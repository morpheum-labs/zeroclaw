//! Long-running hand / memory health checks (filesystem + prompt probe).

use crate::config::Config;
use crate::memory::layered_paths::auto_memory_index_path;
use anyhow::{Context, Result};
use std::path::Path;
use std::time::{Duration, SystemTime};

const STALE: Duration = Duration::from_secs(86_400);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Score {
    Green,
    Yellow,
    Red,
}

fn bump_score(current: Score, bump: Score) -> Score {
    match (current, bump) {
        (Score::Red, _) | (_, Score::Red) => Score::Red,
        (Score::Yellow, _) | (_, Score::Yellow) => Score::Yellow,
        _ => Score::Green,
    }
}

fn age_secs(mtime: SystemTime) -> Option<u64> {
    SystemTime::now()
        .duration_since(mtime)
        .ok()
        .map(|d| d.as_secs())
}

fn fmt_age(secs: Option<u64>) -> String {
    match secs {
        None => "unknown".into(),
        Some(s) if s < 120 => format!("{s}s"),
        Some(s) if s < 7200 => format!("{}m", s / 60),
        Some(s) => format!("{}h", s / 3600),
    }
}

fn file_mtime(path: &Path) -> Option<SystemTime> {
    std::fs::metadata(path).ok()?.modified().ok()
}

/// `zeroclaw doctor long-run` — scratchpad + AutoMemory index + optional hand prompt boundary probe.
pub async fn run(config: &Config, hand_filter: Option<&str>) -> Result<()> {
    let zdir = crate::context::default_user_zeroclaw_dir()
        .context("Could not resolve ~/.zeroclaw (HOME unset?)")?;
    let hands_dir = zdir.join("hands");
    let hands = crate::hands::load_hands(&hands_dir)?;

    let mut selected: Vec<&crate::hands::Hand> = hands.iter().collect();
    if let Some(name) = hand_filter.map(str::trim).filter(|s| !s.is_empty()) {
        selected.retain(|h| h.name == name);
        if selected.is_empty() {
            anyhow::bail!("No hand named `{name}` under {}", hands_dir.display());
        }
    }

    if selected.is_empty() {
        println!("No hands found under {}.", hands_dir.display());
        return Ok(());
    }

    println!("🩺 ZeroClaw Doctor — long-run health");
    println!("  Workspace: {}", config.workspace_dir.display());
    println!();

    let idx_path = auto_memory_index_path(&config.workspace_dir);
    let idx_mtime = file_mtime(&idx_path);
    let idx_age = idx_mtime.and_then(age_secs);
    let idx_stale = idx_age.is_some_and(|s| s >= STALE.as_secs());

    for hand in selected {
        let mut score = Score::Green;
        let mut rec: Vec<String> = Vec::new();

        println!("  [hand: {}]", hand.name);
        println!("    coordinator_mode: {:?}", hand.coordinator_mode);

        let scratch = crate::hands::scratchpad_dir_for_hand(&zdir, &hand.name);
        let decisions = scratch.join("decisions.md");
        let final_sum = scratch.join("final_summary.md");
        let dm = file_mtime(&decisions);
        let fm = file_mtime(&final_sum);
        println!(
            "    scratchpad: {}",
            if scratch.is_dir() {
                scratch.display().to_string()
            } else {
                "(missing)".into()
            }
        );
        if let Some(t) = dm {
            let a = age_secs(t);
            println!("      decisions.md mtime age: {}", fmt_age(a));
            if !matches!(
                hand.coordinator_mode,
                crate::hands::CoordinatorMode::Disabled
            ) {
                if a.map(|s| s > 86_400).unwrap_or(false) {
                    score = bump_score(score, Score::Yellow);
                    rec.push(
                        "Scratchpad decisions look stale for an active coordinator hand.".into(),
                    );
                }
            }
        } else if matches!(
            hand.coordinator_mode,
            crate::hands::CoordinatorMode::Enabled
                | crate::hands::CoordinatorMode::ResearchOnly
                | crate::hands::CoordinatorMode::ExecutionOnly
        ) {
            score = bump_score(score, Score::Yellow);
            rec.push("Coordinator enabled but decisions.md not found yet.".into());
        }

        if fm.is_some() {
            println!(
                "      final_summary.md mtime age: {}",
                fmt_age(fm.and_then(age_secs))
            );
        }

        if config.memory.layered.enabled {
            println!(
                "    layered_memory: enabled | AutoMemory index: {} (age {})",
                idx_path.display(),
                fmt_age(idx_age)
            );
            if idx_stale {
                score = bump_score(score, Score::Yellow);
                rec.push("AutoMemory MEMORY.md index older than 24h.".into());
            }
        } else {
            println!("    layered_memory: disabled (index n/a)");
        }

        let boundary_ok =
            crate::hands::coordinator::probe_hand_prompt_cache_boundary(config, hand).await;
        match boundary_ok {
            Ok(true) => println!("    static/dynamic boundary: present in assembled hand prompt"),
            Ok(false) => {
                score = bump_score(score, Score::Red);
                println!("    static/dynamic boundary: MISSING in assembled hand prompt");
                rec.push("Assembled prompt lacks __SYSTEM_PROMPT_DYNAMIC_BOUNDARY__ — Phase 1 cache split may be broken.".into());
            }
            Err(e) => {
                score = bump_score(score, Score::Yellow);
                println!("    static/dynamic boundary: probe failed ({e:#})");
            }
        }

        let (label, icon) = match score {
            Score::Green => ("green", "✅"),
            Score::Yellow => ("yellow", "⚠️ "),
            Score::Red => ("red", "❌"),
        };
        println!("    overall: {icon} {label}");
        if rec.is_empty() {
            println!("      recommendation: no issues flagged for this snapshot.");
        } else {
            for r in &rec {
                println!("      recommendation: {r}");
            }
        }
        println!();
    }

    Ok(())
}
