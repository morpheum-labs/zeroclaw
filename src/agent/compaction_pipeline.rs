//! Phased compaction before LLM calls and reactive trimming after context-related failures
//! (HTTP 400 / 413-style errors surface through [`llm_error_suggests_context_retry`]).
//!
//! ## Lesson 04 pipeline (conceptual order)
//!
//! 1. **tool_result_budget** — large tool payloads are offloaded in the tool loop
//!    ([`crate::agent::tool_result_offload`]), not in this module.
//! 2. **snipCompact** → **microcompact** → **contextCollapse** → **autoCompact** — today's
//!    [`history_pruner::prune_history`] performs the active trim in one pass; the stage
//!    names are traced here so diagnostics stay aligned with the article as we split
//!    aggressiveness per stage later.
//! 3. **inject_session_memory_and_index** — after prune, build a short markdown block from
//!    the latest session-memory digest and AutoMemory `MEMORY.md` index for dynamic-tail merge.

use crate::agent::context_analyzer;
use crate::agent::history_pruner::{self, HistoryPrunerConfig};
use crate::memory::layered_paths::auto_memory_index_path;
use crate::providers::ChatMessage;
use anyhow::Result;
use std::io::Read;
use std::path::Path;
use std::time::{Duration, SystemTime};

const STALE_SECS: u64 = 86_400;

/// Named compaction stages (article / lesson 04 order + memory injection).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactionStage {
    ToolResultBudget,
    SnipCompact,
    MicroCompact,
    ContextCollapse,
    AutoCompact,
    InjectSessionMemoryAndIndex,
}

/// Ordered stages for logging and future per-stage tuning.
#[must_use]
pub const fn compaction_stage_pipeline() -> [CompactionStage; 6] {
    [
        CompactionStage::ToolResultBudget,
        CompactionStage::SnipCompact,
        CompactionStage::MicroCompact,
        CompactionStage::ContextCollapse,
        CompactionStage::AutoCompact,
        CompactionStage::InjectSessionMemoryAndIndex,
    ]
}

/// Why a compaction pass is running (drives aggressiveness).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CompactionTrigger {
    #[default]
    Routine,
    /// Prior LLM attempt failed with a context-size style error.
    ReactiveContextError,
}

/// Per-iteration context for optional analyzer stages.
#[derive(Debug, Clone)]
pub struct CompactionContext {
    pub iteration: usize,
    pub last_tool_names: Vec<String>,
    pub trigger: CompactionTrigger,
    /// When true, run `analyze_turn_context` to log suggested tools (no filtering yet).
    pub log_context_signals: bool,
    /// Workspace root for layered AutoMemory index reads (optional).
    pub workspace_dir: Option<std::path::PathBuf>,
}

impl CompactionContext {
    #[must_use]
    pub fn new(
        iteration: usize,
        last_tool_names: Vec<String>,
        trigger: CompactionTrigger,
        workspace_dir: Option<std::path::PathBuf>,
    ) -> Self {
        Self {
            iteration,
            last_tool_names,
            trigger,
            log_context_signals: false,
            workspace_dir,
        }
    }
}

/// Build markdown merged into the system prompt dynamic tail (after prune, before assembly).
#[must_use]
pub fn build_memory_injection_fragment(workspace_dir: Option<&Path>) -> Option<String> {
    let mut blocks: Vec<String> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();
    let now = SystemTime::now();

    if let Some(s) = crate::agent::query_engine::peek_session_memory_summary() {
        if let Ok(age) = now.duration_since(s.updated_at) {
            if age > Duration::from_secs(STALE_SECS) {
                warnings.push(format!(
                    "**Warning:** Session memory summary is stale (~{} h > 24 h).",
                    age.as_secs() / 3600
                ));
            }
        }
        if !s.summary_text.trim().is_empty() {
            blocks.push(format!(
                "### Session memory (latest)\n\n{}",
                s.summary_text.trim()
            ));
        }
    }

    if let Some(ws) = workspace_dir {
        let idx_path = auto_memory_index_path(ws);
        let mut index_snippet = String::new();
        let mut index_stale = false;
        if let Ok(meta) = std::fs::metadata(&idx_path) {
            if let Ok(mtime) = meta.modified() {
                if now.duration_since(mtime).ok() > Some(Duration::from_secs(STALE_SECS)) {
                    index_stale = true;
                }
            }
        }
        if let Ok(mut f) = std::fs::File::open(&idx_path) {
            let mut buf = String::new();
            if f.read_to_string(&mut buf).is_ok() && !buf.trim().is_empty() {
                let cap = 6000usize;
                index_snippet = if buf.len() > cap {
                    let end = buf
                        .char_indices()
                        .map(|(i, _)| i)
                        .take_while(|&i| i <= cap)
                        .last()
                        .unwrap_or(0);
                    format!("{}…", &buf[..end])
                } else {
                    buf
                };
            }
        }
        if index_stale {
            warnings.push(
                "**Warning:** AutoMemory MEMORY.md index may be stale (> 24 h since mtime).".into(),
            );
        }
        if !index_snippet.trim().is_empty() {
            blocks.push(format!(
                "### AutoMemory index (`{}`)\n\n{}",
                idx_path.display(),
                index_snippet.trim()
            ));
        }
    }

    if blocks.is_empty() && warnings.is_empty() {
        return None;
    }

    let mut out = String::from("## Memory reload (post-compaction)\n\n");
    if !warnings.is_empty() {
        out.push_str(&warnings.join("\n\n"));
        out.push_str("\n\n");
    }
    out.push_str(&blocks.join("\n\n"));
    Some(out)
}

/// Run pruning (and optional context analysis) before an LLM call; returns optional memory block.
pub fn run_pre_llm_phases(
    history: &mut Vec<ChatMessage>,
    pruning: &HistoryPrunerConfig,
    ctx: &CompactionContext,
) -> Result<Option<String>> {
    for stage in compaction_stage_pipeline() {
        match stage {
            CompactionStage::ToolResultBudget => {
                tracing::trace!(
                    stage = ?stage,
                    iteration = ctx.iteration,
                    "compaction: tool_result_budget (handled in tool loop / offload)"
                );
            }
            CompactionStage::SnipCompact
            | CompactionStage::MicroCompact
            | CompactionStage::ContextCollapse
            | CompactionStage::AutoCompact => {
                tracing::trace!(
                    stage = ?stage,
                    iteration = ctx.iteration,
                    trigger = ?ctx.trigger,
                    "compaction: stage (snip/micro/collapse/auto share one prune pass today)"
                );
            }
            CompactionStage::InjectSessionMemoryAndIndex => {
                tracing::trace!(
                    stage = ?stage,
                    iteration = ctx.iteration,
                    "compaction: inject_session_memory_and_index (runs after prune)"
                );
            }
        }
    }
    if pruning.enabled {
        let _stats = history_pruner::prune_history(history, pruning);
    }

    if ctx.log_context_signals {
        let signals = context_analyzer::analyze_turn_context(
            history,
            "",
            ctx.iteration,
            &ctx.last_tool_names,
        );
        tracing::debug!(
            iteration = ctx.iteration,
            suggested_tools = ?signals.suggested_tools,
            history_relevant = signals.history_relevant,
            "context_analyzer signals"
        );
    }

    let injection = build_memory_injection_fragment(ctx.workspace_dir.as_deref());
    if injection.is_some() {
        crate::agent::query_engine::record_last_memory_injection_now();
        tracing::debug!(
            iteration = ctx.iteration,
            chars = injection.as_ref().map(|s| s.len()).unwrap_or(0),
            "compaction: memory injection fragment prepared"
        );
    }

    Ok(injection)
}

/// Aggressive trim after a context-related LLM failure (best-effort).
pub fn run_reactive_compaction(
    history: &mut Vec<ChatMessage>,
    pruning: &HistoryPrunerConfig,
) -> Result<()> {
    if pruning.enabled {
        let _stats = history_pruner::prune_history(history, pruning);
    }
    Ok(())
}

/// True when the error likely reflects prompt/context limits recoverable by trimming.
#[must_use]
pub fn llm_error_suggests_context_retry(err: &anyhow::Error) -> bool {
    crate::providers::reliable::error_suggests_reactive_compaction(err)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llm_error_suggests_context_retry_detects_window_hints() {
        let e = anyhow::anyhow!("prompt exceeds max length for this model");
        assert!(llm_error_suggests_context_retry(&e));
    }

    #[test]
    fn run_pre_llm_phases_no_panic_when_disabled() {
        let mut history = vec![
            ChatMessage::system("sys"),
            ChatMessage::user("hi".repeat(100)),
        ];
        let pruning = HistoryPrunerConfig::default();
        let ctx = CompactionContext::new(0, vec![], CompactionTrigger::Routine, None);
        run_pre_llm_phases(&mut history, &pruning, &ctx).unwrap();
    }
}
