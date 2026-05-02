//! QueryEngine: diagnostics + traced delegation into [`super::loop_::run_tool_call_loop_body`].
//!
//! This module is the canonical orchestration boundary (lesson 04-style): one turn runs inside
//! [`run_query_loop`], which records transitions and runs post-turn stop hooks.

use super::state::{EngineState, TransitionReason, TurnTransition};
use super::TurnEventSink;
use crate::approval::ApprovalManager;
use crate::hooks::{HookResult, HookRunner};
use crate::memory::consolidation::ConsolidationResult;
use crate::observability::Observer;
use crate::providers::{ChatMessage, Provider};
use crate::tools::Tool;
use anyhow::Result;
use std::collections::VecDeque;
use std::sync::{Arc, LazyLock, Mutex};
use tokio_util::sync::CancellationToken;

const DIAG_CAP: usize = 64;

#[derive(Debug, Clone)]
struct DiagEntry {
    pub ts: std::time::Instant,
    pub transition: TurnTransition,
}

static QUERY_ENGINE_DIAG: LazyLock<Mutex<VecDeque<DiagEntry>>> =
    LazyLock::new(|| Mutex::new(VecDeque::with_capacity(DIAG_CAP)));

pub fn record_transition(reason: TransitionReason, detail: Option<String>) {
    let mut q = QUERY_ENGINE_DIAG.lock().unwrap_or_else(|p| p.into_inner());
    if q.len() >= DIAG_CAP {
        q.pop_front();
    }
    q.push_back(DiagEntry {
        ts: std::time::Instant::now(),
        transition: TurnTransition { reason, detail },
    });
}

/// Recent transitions for `zeroclaw doctor query-engine`.
#[must_use]
pub fn drain_diagnostics() -> Vec<TurnTransition> {
    QUERY_ENGINE_DIAG
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .iter()
        .map(|e| e.transition.clone())
        .collect()
}

/// Clone recent transitions without mutating the deque (read-only diagnostics).
#[must_use]
pub fn peek_diagnostics() -> Vec<TurnTransition> {
    QUERY_ENGINE_DIAG
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .iter()
        .map(|e| e.transition.clone())
        .collect()
}

/// Last system prompt assembly stats (in-process), for `zeroclaw doctor query-engine`.
#[derive(Debug, Clone, Copy, Default)]
pub struct SystemPromptAssemblyDiag {
    pub static_tokens_est: u32,
    pub dynamic_tokens_est: u32,
    pub static_prefix_cached: bool,
}

static LAST_SYSTEM_PROMPT_ASSEMBLY: LazyLock<Mutex<Option<SystemPromptAssemblyDiag>>> =
    LazyLock::new(|| Mutex::new(None));

/// Record approximate token counts (`chars / 4`) and whether the memoized static prefix was reused.
pub fn record_system_prompt_assembly(
    static_tokens_est: u32,
    dynamic_tokens_est: u32,
    static_prefix_cached: bool,
) {
    tracing::info!(
        static_tokens_est,
        dynamic_tokens_est,
        static_prefix_cached,
        "System prompt assembled — static approx tokens, dynamic approx tokens, static prefix memo hit"
    );
    let mut g = LAST_SYSTEM_PROMPT_ASSEMBLY
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    *g = Some(SystemPromptAssemblyDiag {
        static_tokens_est,
        dynamic_tokens_est,
        static_prefix_cached,
    });
}

#[must_use]
pub fn last_system_prompt_assembly() -> Option<SystemPromptAssemblyDiag> {
    LAST_SYSTEM_PROMPT_ASSEMBLY
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .clone()
}

/// Last layered memory selector stats (in-process), for `zeroclaw doctor query-engine`.
#[derive(Debug, Clone, Copy, Default)]
pub struct LayeredMemoryDiag {
    pub topics_picked: usize,
    pub session_injected: bool,
    pub staleness_warnings: usize,
}

static LAST_LAYERED_MEMORY: LazyLock<Mutex<Option<LayeredMemoryDiag>>> =
    LazyLock::new(|| Mutex::new(None));

pub fn record_layered_memory_selection(
    topics_picked: usize,
    session_injected: bool,
    staleness_warnings: usize,
) {
    let mut g = LAST_LAYERED_MEMORY
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    *g = Some(LayeredMemoryDiag {
        topics_picked,
        session_injected,
        staleness_warnings,
    });
}

#[must_use]
pub fn last_layered_memory_selection() -> Option<LayeredMemoryDiag> {
    LAST_LAYERED_MEMORY
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .clone()
}

/// Latest session-memory digest for compaction injection and diagnostics (~300 token budget).
#[derive(Debug, Clone)]
pub struct SessionMemorySummary {
    pub summary_text: String,
    pub updated_at: std::time::SystemTime,
}

static LAST_SESSION_MEMORY_SUMMARY: LazyLock<Mutex<Option<SessionMemorySummary>>> =
    LazyLock::new(|| Mutex::new(None));

static LAST_MEMORY_INJECTION: LazyLock<Mutex<Option<std::time::Instant>>> =
    LazyLock::new(|| Mutex::new(None));

/// Update in-process session summary from a consolidation result (called after awaited consolidation).
pub fn record_session_memory_from_consolidation(r: &ConsolidationResult) {
    let mut parts: Vec<String> = Vec::new();
    let he = r.history_entry.trim();
    if !he.is_empty() {
        parts.push(format!("**Turn:** {he}"));
    }
    for f in r.facts.iter().take(6) {
        let t = f.trim();
        if !t.is_empty() {
            parts.push(format!("- {t}"));
        }
    }
    if let Some(ref t) = r.trend {
        let t = t.trim();
        if !t.is_empty() {
            parts.push(format!("**Trend:** {t}"));
        }
    }
    if let Some(ref u) = r.memory_update {
        let u = u.trim();
        if !u.is_empty() {
            parts.push(format!("**Memory:** {u}"));
        }
    }
    let mut text = parts.join("\n");
    const MAX_CHARS: usize = 1200;
    if text.len() > MAX_CHARS {
        let end = text
            .char_indices()
            .map(|(i, _)| i)
            .take_while(|&i| i <= MAX_CHARS)
            .last()
            .unwrap_or(0);
        text = format!("{}…", &text[..end]);
    }
    let summary = SessionMemorySummary {
        summary_text: text,
        updated_at: std::time::SystemTime::now(),
    };
    let mut g = LAST_SESSION_MEMORY_SUMMARY
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    *g = Some(summary);
}

#[must_use]
pub fn peek_session_memory_summary() -> Option<SessionMemorySummary> {
    LAST_SESSION_MEMORY_SUMMARY
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .clone()
}

pub fn record_last_memory_injection_now() {
    *LAST_MEMORY_INJECTION
        .lock()
        .unwrap_or_else(|p| p.into_inner()) = Some(std::time::Instant::now());
}

#[must_use]
pub fn last_memory_injection() -> Option<std::time::Instant> {
    LAST_MEMORY_INJECTION
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .clone()
}

async fn run_engine_post_turn_consolidation(
    provider: &dyn Provider,
    model: &str,
    memory: &Arc<dyn crate::memory::traits::Memory>,
    user_message: &str,
    assistant_summary: &str,
) {
    match crate::memory::consolidation::consolidate_turn(
        provider,
        model,
        memory.as_ref(),
        user_message,
        assistant_summary,
    )
    .await
    {
        Ok(r) => record_session_memory_from_consolidation(&r),
        Err(e) => tracing::debug!(error = %e, "post-turn memory consolidation failed"),
    }
}

/// Heuristic: model may have stopped early due to output token cap — caller may append a nudge.
#[must_use]
pub fn should_request_token_continuation(
    usage: Option<&crate::providers::traits::TokenUsage>,
    output_text_chars: usize,
) -> bool {
    let Some(u) = usage else {
        return false;
    };
    let Some(out) = u.output_tokens else {
        return false;
    };
    // Without provider-reported max_output_tokens, use a conservative heuristic:
    // large billed output with almost no visible text often indicates truncation.
    out >= 900 && output_text_chars < 24
}

/// Article-style outer turn loop: diagnostics + body + post-turn hooks.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_query_loop(
    state: &mut EngineState,
    provider: &dyn Provider,
    history: &mut Vec<ChatMessage>,
    tools_registry: &[Box<dyn Tool>],
    observer: &dyn Observer,
    provider_name: &str,
    model: &str,
    temperature: f64,
    silent: bool,
    approval: Option<&ApprovalManager>,
    channel_name: &str,
    channel_reply_target: Option<&str>,
    multimodal_config: &crate::config::MultimodalConfig,
    max_tool_iterations: usize,
    cancellation_token: Option<CancellationToken>,
    turn_event_sink: Option<tokio::sync::mpsc::Sender<TurnEventSink>>,
    hooks: Option<&HookRunner>,
    excluded_tools: &[String],
    dedup_exempt_tools: &[String],
    activated_tools: Option<&std::sync::Arc<std::sync::Mutex<crate::tools::ActivatedToolSet>>>,
    model_switch_callback: Option<super::loop_::ModelSwitchCallback>,
    pacing: &crate::config::PacingConfig,
    tool_result_offload: &crate::config::ToolResultOffloadConfig,
    history_pruning: &crate::agent::history_pruner::HistoryPrunerConfig,
    turn_user_message: Option<&str>,
    system_prompt_refresh: Option<&super::system_prompt::SystemPromptAssemblyRefs<'_>>,
    post_turn_memory: super::loop_::PostTurnMemoryBinding,
) -> Result<String> {
    state.last_transition = Some(TransitionReason::BeginTurn);
    record_transition(TransitionReason::BeginTurn, None);
    let res = super::loop_::run_tool_call_loop_body(
        provider,
        history,
        tools_registry,
        observer,
        provider_name,
        model,
        temperature,
        silent,
        approval,
        channel_name,
        channel_reply_target,
        multimodal_config,
        max_tool_iterations,
        cancellation_token,
        turn_event_sink,
        hooks,
        excluded_tools,
        dedup_exempt_tools,
        activated_tools,
        model_switch_callback,
        pacing,
        tool_result_offload,
        history_pruning,
        system_prompt_refresh,
    )
    .await;
    match &res {
        Ok(_) => {
            state.last_transition = Some(TransitionReason::TurnComplete);
            record_transition(TransitionReason::TurnComplete, None);
        }
        Err(e) => {
            state.last_transition = Some(TransitionReason::TurnError);
            record_transition(TransitionReason::TurnError, Some(e.to_string()));
        }
    }
    if let Ok(text) = &res {
        if post_turn_memory.auto_save {
            if let Some(mem) = &post_turn_memory.memory {
                let user = turn_user_message.unwrap_or("");
                run_engine_post_turn_consolidation(provider, model, mem, user, text.as_str()).await;
            }
        }
    }
    if let (Ok(text), Some(hooks)) = (&res, hooks) {
        let user = turn_user_message.unwrap_or("");
        super::stop_hooks::fire_after_turn_void(hooks, channel_name, user, text.as_str()).await;
        match super::stop_hooks::run_after_turn_blocking(hooks, channel_name, user, text.as_str())
            .await
        {
            HookResult::Continue(()) => {}
            HookResult::Cancel(reason) => {
                record_transition(TransitionReason::StopHookBlocking, Some(reason));
            }
        }
    }
    res
}

// ── Hand coordinator: worker fork + last-line diagnostics ─────────────────

static LAST_COORDINATOR_SUMMARY: LazyLock<Mutex<Option<String>>> =
    LazyLock::new(|| Mutex::new(None));

/// Last coordinator status line for `zeroclaw doctor query-engine`.
pub fn set_last_coordinator_summary(summary: Option<String>) {
    *LAST_COORDINATOR_SUMMARY
        .lock()
        .unwrap_or_else(|p| p.into_inner()) = summary;
}

#[must_use]
pub fn last_coordinator_summary() -> Option<String> {
    LAST_COORDINATOR_SUMMARY
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .clone()
}

/// Outcome from one forked worker [`run_worker_fork`] sub-loop.
#[derive(Debug, Clone)]
pub struct WorkerForkOutcome {
    pub final_text: String,
}

/// Run a short-lived worker with the same provider as the parent agent, excluding every
/// tool not permitted for this worker and not in the hand allowlist (via loop exclusions).
///
/// `system_prompt` is the full system message for this worker (typically worker-scoped assembly
/// for coordinator phases, or the full hand assembly for single-turn `Disabled` mode).
#[allow(clippy::too_many_arguments)]
pub async fn run_worker_fork(
    cfg: &crate::config::Config,
    provider: &dyn Provider,
    provider_name: &str,
    model: &str,
    temperature: f64,
    system_prompt: &super::system_prompt::SystemPrompt,
    parent_summary: &str,
    worker_goal: &str,
    worker_tool_names: &[String],
    // When `Some`, only these tool names may run (hand TOML allowlist).
    hand_allowed_tools: Option<&[String]>,
    tools_registry: &[Box<dyn Tool>],
    observer: &dyn Observer,
    hand_name: &str,
    phase: &str,
    max_tool_iterations: usize,
    memory: Arc<dyn crate::memory::traits::Memory>,
) -> Result<WorkerForkOutcome> {
    use std::collections::HashSet;

    let worker_allowed: HashSet<&str> = worker_tool_names.iter().map(String::as_str).collect();

    fn tool_ok_for_hand(allowed: Option<&[String]>, tool_name: &str) -> bool {
        match allowed {
            None => true,
            Some(list) => list.iter().any(|x| x == tool_name),
        }
    }

    let mut excluded: Vec<String> = tools_registry
        .iter()
        .map(|t| t.name().to_string())
        .filter(|n| {
            let n = n.as_str();
            !tool_ok_for_hand(hand_allowed_tools, n) || !worker_allowed.contains(n)
        })
        .collect();

    if !worker_allowed.contains("delegate")
        && tools_registry.iter().any(|t| t.name() == "delegate")
        && !excluded.iter().any(|e| e == "delegate")
    {
        excluded.push("delegate".into());
    }

    record_transition(
        TransitionReason::CoordinatorWorkerSpawn,
        Some(format!(
            "hand={hand_name} phase={phase} allow=[{}]",
            worker_tool_names.join(",")
        )),
    );
    set_last_coordinator_summary(Some(format!(
        "Coordinator: worker `{phase}` starting (hand `{hand_name}`)"
    )));

    let mut history = super::system_prompt::build_forked_history(
        system_prompt,
        parent_summary,
        worker_goal,
        worker_tool_names,
    );

    if cfg.memory.layered.enabled && cfg.memory.auto_save {
        crate::memory::layered_context::install_pending_layered_turn(Some(
            crate::memory::layered_context::LayeredTurnContext {
                workspace_dir: cfg.workspace_dir.clone(),
                session_key: format!("hand:{hand_name}"),
                layered: cfg.memory.layered.clone(),
            },
        ));
    } else {
        crate::memory::layered_context::install_pending_layered_turn(None);
    }

    let post_turn = super::loop_::PostTurnMemoryBinding {
        memory: Some(Arc::clone(&memory)),
        auto_save: cfg.memory.auto_save,
    };

    let res = super::loop_::run_tool_call_loop(
        provider,
        &mut history,
        tools_registry,
        observer,
        provider_name,
        model,
        temperature,
        true,
        None,
        "hands:coordinator",
        None,
        &cfg.multimodal,
        max_tool_iterations,
        None,
        None,
        None,
        &excluded,
        &[],
        None,
        None,
        &cfg.pacing,
        &cfg.agent.tool_result_offload,
        &cfg.agent.history_pruning,
        Some(worker_goal),
        None,
        post_turn,
    )
    .await;

    match &res {
        Ok(text) => {
            record_transition(
                TransitionReason::CoordinatorWorkerComplete,
                Some(format!(
                    "hand={hand_name} phase={phase} ok chars={}",
                    text.len()
                )),
            );
            set_last_coordinator_summary(Some(format!(
                "Coordinator: worker `{phase}` completed ({} chars)",
                text.len()
            )));
        }
        Err(e) => {
            record_transition(
                TransitionReason::CoordinatorWorkerComplete,
                Some(format!("hand={hand_name} phase={phase} err={e}")),
            );
            set_last_coordinator_summary(Some(format!(
                "Coordinator: worker `{phase}` failed: {e}"
            )));
        }
    }

    res.map(|final_text| WorkerForkOutcome { final_text })
}
