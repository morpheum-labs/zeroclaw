//! Coordinator mode for hands: deterministic phased workers, forked prompt context, scratchpad.

use std::collections::HashSet;
use std::fmt::Write as _;
use std::path::Path;

use anyhow::{bail, Context, Result};

use crate::agent::system_prompt::{assemble_once, PromptAssemblyContext, SystemPrompt};
use crate::agent::{query_engine, Agent};
use crate::config::Config;
use crate::context::DynamicContextPaths;
use crate::security::SecurityPolicy;

use super::load_hand_context;
use super::types::{CoordinatorMode, Hand, HandContext};
use super::{ensure_scratchpad_dir, scratchpad_dir_for_hand};

fn append_decision(scratchpad: &Path, line: &str) -> Result<()> {
    let path = scratchpad.join("decisions.md");
    let stamp = chrono::Utc::now().to_rfc3339();
    let mut prev = if path.exists() {
        std::fs::read_to_string(&path).unwrap_or_default()
    } else {
        String::new()
    };
    if !prev.is_empty() && !prev.ends_with('\n') {
        prev.push('\n');
    }
    let _ = writeln!(&mut prev, "- [{stamp}] {line}");
    std::fs::write(&path, prev).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut end = max;
    while !s.is_char_boundary(end) && end > 0 {
        end -= 1;
    }
    format!("{}… [truncated]", &s[..end])
}

/// Scratchpad-first handoff: no inlined prior worker bodies (they live under `scratchpad/`).
fn scratchpad_parent_summary(
    hand: &Hand,
    scratchpad: &Path,
    ctx: &HandContext,
    completed_rel_paths: &[&str],
) -> String {
    let sp = scratchpad.display().to_string();
    let mut s = format!(
        "Hand: {}\nDescription: {}\nMission: {}\n\n## Scratchpad (on disk — use `file_read` on these paths; prior phase outputs are not pasted here)\n\nDirectory: `{sp}`\n",
        hand.name, hand.description, hand.prompt
    );
    for rel in completed_rel_paths {
        let p = scratchpad.join(rel);
        if p.exists() {
            if let Ok(meta) = std::fs::metadata(&p) {
                let _ = writeln!(&mut s, "- `{}/{}` ({} bytes)", sp, rel, meta.len());
            } else {
                let _ = writeln!(&mut s, "- `{sp}/{rel}`");
            }
        }
    }
    s.push_str("\n## Rolling context (compact)\n\n");
    s.push_str(&truncate(&context_digest(ctx), 1200));
    s
}

/// `STATUS: PASS` / `STATUS: FAIL` (case-insensitive); last matching line in the text wins.
fn line_verification_status(line: &str) -> Option<bool> {
    let t = line.trim();
    let (head, tail) = t.split_once(':')?;
    if !head.trim().eq_ignore_ascii_case("STATUS") {
        return None;
    }
    match tail.trim().to_ascii_uppercase().as_str() {
        "PASS" => Some(true),
        "FAIL" => Some(false),
        _ => None,
    }
}

fn scan_verification_status(text: &str) -> Option<bool> {
    for line in text.lines().rev() {
        if let Some(v) = line_verification_status(line) {
            return Some(v);
        }
    }
    None
}

/// Default tool candidates per coordinator phase (intersected with the hand allowlist and registry).
fn phase_default_tool_names(phase: &str) -> &'static [&'static str] {
    match phase {
        "research" => &[
            "file_read",
            "memory_recall",
            "memory_store",
            "web_search_tool",
            "http_request",
            "web_fetch",
        ],
        "synthesis" => &["file_read", "memory_recall", "file_write", "memory_store"],
        "implementation" => &["file_read", "file_write", "shell", "memory_store"],
        "verification" => &["file_read", "shell", "memory_recall"],
        _ => &["file_read"],
    }
}

fn pick_worker_tool_names(
    phase: &str,
    hand: &Hand,
    registry: &[Box<dyn crate::tools::Tool>],
) -> Vec<String> {
    let reg: HashSet<&str> = registry.iter().map(|t| t.name()).collect();
    let mut names: Vec<String> = phase_default_tool_names(phase)
        .iter()
        .filter(|n| reg.contains(*n))
        .map(|s| (*s).to_string())
        .collect();

    if let Some(allowed) = hand.allowed_tools.as_ref() {
        let allow: HashSet<&str> = allowed.iter().map(String::as_str).collect();
        names.retain(|n| allow.contains(n.as_str()));
    }

    if names.is_empty() {
        if reg.contains("file_read") {
            if tool_ok_for_hand(hand.allowed_tools.as_deref(), "file_read") {
                names.push("file_read".into());
            }
        }
    }
    names
}

fn tool_ok_for_hand(allowed: Option<&[String]>, tool_name: &str) -> bool {
    match allowed {
        None => true,
        Some(list) => list.iter().any(|x| x == tool_name),
    }
}

fn hand_allow_slice(hand: &Hand) -> Option<&[String]> {
    hand.allowed_tools.as_ref().map(Vec::as_slice)
}

fn all_hand_tool_names(hand: &Hand, registry: &[Box<dyn crate::tools::Tool>]) -> Vec<String> {
    registry
        .iter()
        .map(|t| t.name().to_string())
        .filter(|n| tool_ok_for_hand(hand_allow_slice(hand), n))
        .collect()
}

/// Assemble system prompt for a hand, optionally restricted to a subset of tool names (worker phase).
fn assemble_hand_system_prompt(
    config: &Config,
    agent: &Agent,
    hand: &Hand,
    only_tool_names: Option<&[String]>,
) -> Result<SystemPrompt> {
    let security = SecurityPolicy::from_config(&config.autonomy, &config.workspace_dir);
    let security_summary = security.prompt_summary();
    let skills = crate::skills::load_skills_with_config(&config.workspace_dir, config);
    let user_zc = crate::context::default_user_zeroclaw_dir();
    let dynamic_paths = DynamicContextPaths {
        global_config_dir: None,
        user_config_dir: user_zc.as_deref(),
        session_dir: None,
    };

    let filter: Option<HashSet<&str>> =
        only_tool_names.map(|names| names.iter().map(String::as_str).collect());

    let owned_descriptions: Vec<(String, String)> = agent
        .tools_registry()
        .iter()
        .filter(|t| tool_ok_for_hand(hand_allow_slice(hand), t.name()))
        .filter(|t| filter.as_ref().is_none_or(|set| set.contains(t.name())))
        .map(|t| (t.name().to_string(), t.description().to_string()))
        .collect();
    let pair_refs: Vec<(&str, &str)> = owned_descriptions
        .iter()
        .map(|(a, b)| (a.as_str(), b.as_str()))
        .collect();

    let model = hand
        .model
        .as_deref()
        .unwrap_or_else(|| agent.model_name_str());

    let ctx = PromptAssemblyContext {
        workspace_dir: &config.workspace_dir,
        model_name: model,
        tools: &pair_refs,
        skills: &skills,
        identity_config: Some(&config.identity),
        bootstrap_max_chars: if config.agent.compact_context {
            Some(6000)
        } else {
            None
        },
        autonomy_config: Some(&config.autonomy),
        native_tools: agent.provider_ref().supports_native_tools(),
        skills_prompt_mode: config.skills.prompt_injection_mode,
        compact_context: config.agent.compact_context,
        max_system_prompt_chars: config.agent.max_system_prompt_chars,
        dynamic_context: Some(&config.agent.dynamic_context),
        dynamic_paths,
        static_suffix: "",
        dispatcher_instructions: None,
        security_summary: Some(security_summary.as_str()),
        include_channel_media: false,
        layered_memory_markdown: None,
    };

    assemble_once(&ctx)
}

/// For `zeroclaw doctor long-run`: verify assembled hand system prompt contains the Phase 1 cache boundary marker.
pub async fn probe_hand_prompt_cache_boundary(
    config: &Config,
    hand: &Hand,
) -> anyhow::Result<bool> {
    let agent = crate::agent::Agent::from_config(config).await?;
    let sp = assemble_hand_system_prompt(config, &agent, hand, None)?;
    Ok(sp
        .full()
        .contains(crate::agent::system_prompt::SYSTEM_PROMPT_DYNAMIC_BOUNDARY))
}

fn context_digest(ctx: &HandContext) -> String {
    let mut s = String::new();
    if !ctx.learned_facts.is_empty() {
        let _ = writeln!(&mut s, "### Learned facts\n");
        for f in ctx.learned_facts.iter().take(40) {
            let _ = writeln!(&mut s, "- {f}");
        }
    }
    if let Some(r) = ctx.history.first() {
        let _ = writeln!(
            &mut s,
            "\n### Last run\nstatus: {:?}\nfindings: {:?}",
            r.status, r.findings
        );
    }
    s
}

fn pipeline_phases(mode: CoordinatorMode) -> Vec<(&'static str, &'static str)> {
    match mode {
        CoordinatorMode::Disabled => Vec::new(),
        CoordinatorMode::Enabled => vec![
            ("research", "research.md"),
            ("synthesis", "synthesis.md"),
            ("implementation", "implementation.md"),
            ("verification", "verification.md"),
        ],
        CoordinatorMode::ResearchOnly => {
            vec![("research", "research.md"), ("synthesis", "synthesis.md")]
        }
        CoordinatorMode::ExecutionOnly => vec![
            ("synthesis", "synthesis.md"),
            ("implementation", "implementation.md"),
            ("verification", "verification.md"),
        ],
    }
}

fn worker_goal_for_phase(
    phase: &str,
    rel: &str,
    hand: &Hand,
    scratchpad: &Path,
    ctx: &HandContext,
) -> String {
    let sp = scratchpad.display().to_string();
    let digest = context_digest(ctx);
    match phase {
        "research" => format!(
            "Hand `{}` — research phase.\n\n## Mission\n{}\n\n## Knowledge lines\n{}\n\n## Rolling context\n{digest}\n\nUse `file_read` only if needed. Write durable notes to `{sp}/{rel}` (the coordinator persists your final reply there too).",
            hand.name,
            hand.prompt,
            hand.knowledge.join("\n"),
            digest = digest,
            sp = sp,
            rel = rel,
        ),
        "synthesis" => format!(
            "Synthesis phase for hand `{}`. Read `{sp}/research.md` with `file_read` before writing. Produce a consolidated plan in `{sp}/{rel}`.\n\n{digest}",
            hand.name,
            sp = sp,
            rel = rel,
            digest = digest
        ),
        "implementation" => format!(
            "Implementation phase for hand `{}`. Read `{sp}/synthesis.md` (and other scratchpad files as needed) with `file_read`. Execute concrete steps; document results in `{sp}/{rel}`.\n\n{digest}",
            hand.name,
            sp = sp,
            rel = rel,
            digest = digest
        ),
        "verification" => format!(
            "Verification phase for hand `{}`. Read `{sp}/implementation.md` and related files under `{sp}`. Report pass/fail and risks in `{sp}/{rel}`.\n\nInclude a single final line exactly: `STATUS: PASS` or `STATUS: FAIL`.\n\n{digest}",
            hand.name,
            sp = sp,
            rel = rel,
            digest = digest
        ),
        _ => format!(
            "Worker phase {phase}. Scratchpad: {sp}/{rel}\n\n{digest}",
            sp = sp,
            rel = rel,
            digest = digest,
        ),
    }
}

/// Run one hand using coordinator mode (or a single worker-style turn when [`CoordinatorMode::Disabled`]).
pub async fn run_coordinator_hand(
    config: &Config,
    hands_dir: &Path,
    hand: &Hand,
) -> Result<String> {
    let zdir = crate::context::default_user_zeroclaw_dir()
        .context("HOME / ~/.zeroclaw not available for scratchpad")?;
    let scratchpad = ensure_scratchpad_dir(&zdir, &hand.name)?;
    let hand_ctx = load_hand_context(hands_dir, &hand.name)?;

    let agent = Agent::from_config(config).await?;

    let model = hand
        .model
        .as_deref()
        .unwrap_or_else(|| agent.model_name_str());
    let provider_name = agent.provider_label_str();
    let obs = agent.observer();

    if matches!(hand.coordinator_mode, CoordinatorMode::Disabled) {
        let parent_sp = assemble_hand_system_prompt(config, &agent, hand, None)?;
        query_engine::record_transition(
            crate::agent::state::TransitionReason::CoordinatorModeActive,
            Some(format!("hand={} mode=disabled single_turn", hand.name)),
        );
        query_engine::set_last_coordinator_summary(Some(
            "Coordinator: single-turn hand run (coordinator_mode = disabled)".into(),
        ));
        let names = all_hand_tool_names(hand, agent.tools_registry());
        let goal = format!(
            "{}\n\n## Knowledge\n{}\n\n## Context\n{}",
            hand.prompt,
            hand.knowledge.join("\n"),
            context_digest(&hand_ctx)
        );
        let out = query_engine::run_worker_fork(
            config,
            agent.provider_ref(),
            provider_name,
            model,
            agent.temperature(),
            &parent_sp,
            "(single-turn hand; no prior coordinator summary.)",
            &goal,
            &names,
            hand_allow_slice(hand),
            agent.tools_registry(),
            obs.as_ref(),
            &hand.name,
            "single",
            config.agent.max_tool_iterations,
            agent.memory_handle(),
        )
        .await?;
        append_decision(
            &scratchpad,
            &format!("single_turn ok chars={}", out.final_text.len()),
        )?;
        if let Some(s) = query_engine::peek_session_memory_summary() {
            let digest: String = s.summary_text.chars().take(160).collect();
            if !digest.trim().is_empty() {
                append_decision(&scratchpad, &format!("post_turn_memory_digest: {digest}"))?;
            }
        }
        return Ok(out.final_text);
    }

    query_engine::record_transition(
        crate::agent::state::TransitionReason::CoordinatorModeActive,
        Some(format!(
            "hand={} mode={:?}",
            hand.name, hand.coordinator_mode
        )),
    );
    query_engine::set_last_coordinator_summary(Some(format!(
        "Coordinator mode active — hand `{}` ({:?})",
        hand.name, hand.coordinator_mode
    )));
    append_decision(
        &scratchpad,
        &format!("start coordinator_mode={:?}", hand.coordinator_mode),
    )?;

    let phases = pipeline_phases(hand.coordinator_mode);
    if phases.is_empty() {
        bail!("coordinator pipeline empty");
    }
    let last_rel = phases.last().map(|p| p.1).unwrap_or("verification.md");

    let mut completed_rels: Vec<&'static str> = Vec::new();
    let mut last_out = String::new();

    for (phase, rel) in &phases {
        let tool_names = pick_worker_tool_names(*phase, hand, agent.tools_registry());
        if tool_names.is_empty() {
            bail!(
                "no tools available for phase {} (check hand allowed_tools)",
                phase
            );
        }
        let worker_sp = assemble_hand_system_prompt(config, &agent, hand, Some(&tool_names))?;
        let parent_summary =
            scratchpad_parent_summary(hand, &scratchpad, &hand_ctx, &completed_rels);
        let goal = worker_goal_for_phase(*phase, rel, hand, &scratchpad, &hand_ctx);
        let max_it = (24usize).min(config.agent.max_tool_iterations.max(1));

        let spec_line = format!("phase={} tools={}", phase, tool_names.join(","));
        append_decision(&scratchpad, &spec_line)?;

        let out = match query_engine::run_worker_fork(
            config,
            agent.provider_ref(),
            provider_name,
            model,
            agent.temperature(),
            &worker_sp,
            &parent_summary,
            &goal,
            &tool_names,
            hand_allow_slice(hand),
            agent.tools_registry(),
            obs.as_ref(),
            &hand.name,
            *phase,
            max_it,
            agent.memory_handle(),
        )
        .await
        {
            Ok(o) => o,
            Err(e) => {
                append_decision(
                    &scratchpad,
                    &format!("phase={phase} ERROR pipeline_halt err={e}"),
                )?;
                query_engine::set_last_coordinator_summary(Some(format!(
                    "Coordinator: halted at `{phase}` for hand `{}`: {e}",
                    hand.name
                )));
                return Err(e);
            }
        };

        append_decision(
            &scratchpad,
            &format!(
                "phase={phase} ok chars={} empty={}",
                out.final_text.len(),
                out.final_text.trim().is_empty()
            ),
        )?;

        if let Some(s) = query_engine::peek_session_memory_summary() {
            let digest: String = s.summary_text.chars().take(160).collect();
            if !digest.trim().is_empty() {
                append_decision(&scratchpad, &format!("post_turn_memory_digest: {digest}"))?;
            }
        }

        if *phase == "research" && out.final_text.trim().is_empty() {
            append_decision(&scratchpad, "gate research_empty pipeline_halt")?;
            bail!("research phase produced empty output; see scratchpad/decisions.md");
        }

        if *phase == "verification" {
            match scan_verification_status(&out.final_text) {
                Some(false) => {
                    append_decision(&scratchpad, "gate verification STATUS:FAIL pipeline_halt")?;
                    bail!("verification phase reported STATUS: FAIL; see scratchpad/decisions.md");
                }
                None => {
                    append_decision(
                        &scratchpad,
                        "gate verification missing STATUS line (expected STATUS: PASS or STATUS: FAIL)",
                    )?;
                }
                Some(true) => {
                    append_decision(&scratchpad, "gate verification STATUS:PASS")?;
                }
            }
        }

        let path = scratchpad.join(rel);
        std::fs::write(&path, &out.final_text)
            .with_context(|| format!("failed to write {}", path.display()))?;

        completed_rels.push(rel);
        last_out = out.final_text;
    }

    append_decision(&scratchpad, "coordinator pipeline completed")?;
    query_engine::set_last_coordinator_summary(Some(format!(
        "Coordinator: finished all phases for hand `{}`",
        hand.name
    )));

    let summary_path = scratchpad.join("final_summary.md");
    std::fs::write(&summary_path, &last_out)
        .with_context(|| format!("failed to write {}", summary_path.display()))?;

    Ok(format!(
        "Coordinator finished for hand `{}`. Scratchpad: {}\nLast phase output written to {} and {}.",
        hand.name,
        scratchpad_dir_for_hand(&zdir, &hand.name).display(),
        scratchpad.join(last_rel).display(),
        summary_path.display()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scan_verification_status_pass() {
        let t = "Checks ok.\nSTATUS: PASS\n";
        assert_eq!(scan_verification_status(t), Some(true));
    }

    #[test]
    fn scan_verification_status_fail() {
        let t = "Broken.\nstatus: FAIL\n";
        assert_eq!(scan_verification_status(t), Some(false));
    }

    #[test]
    fn scan_verification_status_none() {
        assert_eq!(scan_verification_status("no status here"), None);
    }

    #[test]
    fn scan_verification_status_ignores_notfail_substring() {
        assert_eq!(
            scan_verification_status("NOTFAIL is not a status line\n"),
            None
        );
    }

    #[test]
    fn scratchpad_parent_summary_lists_completed_files() {
        let tmp = tempfile::tempdir().unwrap();
        let sp = tmp.path().join("scratch");
        std::fs::create_dir_all(&sp).unwrap();
        std::fs::write(sp.join("research.md"), "x").unwrap();
        let hand = Hand {
            name: "h1".into(),
            description: "d".into(),
            schedule: crate::cron::Schedule::Every { every_ms: 60_000 },
            prompt: "do".into(),
            knowledge: vec![],
            allowed_tools: None,
            model: None,
            active: true,
            max_history: 10,
            coordinator_mode: CoordinatorMode::Enabled,
        };
        let ctx = HandContext::new("h1");
        let s = scratchpad_parent_summary(&hand, &sp, &ctx, &["research.md"]);
        assert!(s.contains("Hand: h1"));
        assert!(s.contains("research.md"));
        assert!(s.contains("bytes"));
    }
}
