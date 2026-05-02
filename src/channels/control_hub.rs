//! Telegram (and future channel) **control hub** — pre-LLM slash-style dispatch for
//! safe read-mostly operations and selected `zeroclaw` CLI mirrors.
//!
//! See [docs/design/telegram-control-hub-spec.md](../../../../docs/design/telegram-control-hub-spec.md).

use crate::config::Config;
use anyhow::{Context, Result};
use std::path::PathBuf;
use std::process::Command;

/// Strip a leading `/<prefix>` or `/<prefix>@botname` and return whitespace-split args.
/// `prefix` must not include a leading slash (e.g. `"z"` for `/z skills list`).
pub fn parse_hub_invocation(content: &str, prefix: &str) -> Option<Vec<String>> {
    let trimmed = content.trim();
    if !trimmed.starts_with('/') {
        return None;
    }
    let mut parts = trimmed.split_whitespace();
    let token = parts.next()?;
    let base = token.strip_prefix('/')?.split('@').next()?;
    if base != prefix {
        return None;
    }
    Some(parts.map(|s| s.to_string()).collect())
}

fn hub_prefix_for_telegram(config: &Config) -> Option<&str> {
    let tg = config.channels_config.telegram.as_ref()?;
    if !tg.control_hub_enabled {
        return None;
    }
    let p = tg.control_hub_prefix.trim();
    if p.is_empty() || !is_valid_telegram_command_token(p) {
        return None;
    }
    Some(p)
}

/// `true` when this inbound message should be handled by the control hub (Telegram only).
pub fn telegram_control_hub_should_handle(config: &Config, channel: &str, content: &str) -> bool {
    if channel != "telegram" {
        return false;
    }
    let Some(prefix) = hub_prefix_for_telegram(config) else {
        return false;
    };
    parse_hub_invocation(content, prefix).is_some()
}

fn is_valid_telegram_command_token(s: &str) -> bool {
    let b = s.as_bytes();
    if b.is_empty() || b.len() > 32 {
        return false;
    }
    b.iter().all(|c| c.is_ascii_alphanumeric() || *c == b'_')
}

/// Build `BotCommand.command` values for Telegram `setMyCommands` (no leading slash).
pub fn telegram_bot_command_rows(config: &Config) -> Vec<(String, String)> {
    let mut rows = vec![
        (
            "new".to_string(),
            "Clear this chat's session and start fresh".to_string(),
        ),
        (
            "models".to_string(),
            "List or switch LLM provider for this chat".to_string(),
        ),
        (
            "model".to_string(),
            "Show or set the model id for this chat".to_string(),
        ),
        (
            "config".to_string(),
            "Show routing / model configuration summary".to_string(),
        ),
    ];

    if let Some(prefix) = hub_prefix_for_telegram(config) {
        rows.push((
            prefix.to_string(),
            "Control hub: skills, cron, memory, channel, … (see docs)".to_string(),
        ));
    }

    let skills = crate::skills::load_skills_with_config(&config.workspace_dir, config);
    for s in skills {
        if !s.user_invocable {
            continue;
        }
        let cmd = sanitize_bot_command_token(&s.name);
        if cmd.is_empty() || rows.iter().any(|(c, _)| c == &cmd) {
            continue;
        }
        let mut desc = s.description.clone();
        if desc.chars().count() > 240 {
            desc = desc.chars().take(237).collect::<String>() + "...";
        }
        rows.push((cmd, desc));
    }

    rows
}

fn sanitize_bot_command_token(name: &str) -> String {
    let lower = name.trim().to_ascii_lowercase();
    let mut out = String::new();
    for ch in lower.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            if out.len() < 32 {
                out.push(ch);
            }
        } else if ch == '-' && out.len() < 32 {
            out.push('_');
        }
    }
    out
}

fn zeroclaw_exe() -> PathBuf {
    std::env::current_exe().unwrap_or_else(|_| PathBuf::from("zeroclaw"))
}

/// Run `zeroclaw` with argv (excluding argv[0]); captures merged stdout/stderr.
pub async fn run_zeroclaw_cli(config: &Config, args: Vec<String>) -> Result<String> {
    let exe = zeroclaw_exe();
    let config_path = config.config_path.clone();
    tokio::task::spawn_blocking(move || {
        let mut cmd = Command::new(&exe);
        if let Some(dir) = config_path.parent() {
            cmd.env("ZEROCLAW_CONFIG_DIR", dir);
        }
        cmd.args(&args);
        let out = cmd
            .output()
            .with_context(|| format!("failed to spawn {}", exe.display()))?;
        let mut text = String::new();
        if !out.stdout.is_empty() {
            text.push_str(&String::from_utf8_lossy(&out.stdout));
        }
        if !out.stderr.is_empty() {
            if !text.is_empty() && !text.ends_with('\n') {
                text.push('\n');
            }
            text.push_str(&String::from_utf8_lossy(&out.stderr));
        }
        let code = out.status.code().unwrap_or(-1);
        if !out.status.success() {
            anyhow::bail!("zeroclaw {:?} exited with {code}\n{text}", args);
        }
        Ok(text)
    })
    .await
    .context("spawn_blocking join failed")?
}

/// Dispatch hub argv (already split; e.g. `["skills","list"]`).
pub async fn dispatch_hub(config: &Config, args: &[String]) -> Result<String> {
    if args.is_empty() {
        return Ok(hub_help_text(config));
    }
    match args[0].as_str() {
        "skills" => dispatch_skills(config, &args[1..]).await,
        "mcp" => dispatch_mcp(&args[1..]),
        "shell" => dispatch_shell(config, &args[1..]).await,
        "cron" => dispatch_cron(config, &args[1..]).await,
        "memory" => dispatch_memory(config, &args[1..]).await,
        "hardware" => dispatch_hardware(config, &args[1..]).await,
        "estop" => dispatch_estop(config, &args[1..]).await,
        "channel" => dispatch_channel(config, &args[1..]).await,
        "config" => dispatch_config_schema(config, &args[1..]).await,
        other => anyhow::bail!("Unknown hub topic `{other}`.\n{}", hub_help_text(config)),
    }
}

fn hub_help_text(config: &Config) -> String {
    let prefix = config
        .channels_config
        .telegram
        .as_ref()
        .map(|t| t.control_hub_prefix.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "z".to_string());
    format!(
        "ZeroClaw control hub — send `/{prefix} <topic> …`.\n\
Topics: skills, mcp, shell, cron, memory, hardware, estop, channel, config.\n\
Example: `/{prefix} skills list`, `/{prefix} cron list`.\n\
`mcp serve` is not available here (long-running); use the host CLI.\n\
`estop resume` may require OTP — use `zeroclaw estop` on the host if prompted."
    )
}

async fn dispatch_skills(config: &Config, tail: &[String]) -> Result<String> {
    match tail.first().map(|s| s.as_str()) {
        None | Some("list") => Ok(crate::skills::format_installed_skills_list(config)),
        Some("install") => {
            let Some(src) = tail.get(1) else {
                anyhow::bail!("usage: skills install <url-or-path>");
            };
            run_zeroclaw_cli(config, vec!["skills".into(), "install".into(), src.clone()]).await
        }
        Some("remove") => {
            let Some(name) = tail.get(1) else {
                anyhow::bail!("usage: skills remove <name>");
            };
            run_zeroclaw_cli(config, vec!["skills".into(), "remove".into(), name.clone()]).await
        }
        Some(other) => anyhow::bail!("unknown skills subcommand `{other}`"),
    }
}

fn dispatch_mcp(tail: &[String]) -> Result<String> {
    if tail.first().is_some_and(|s| s == "serve") {
        anyhow::bail!(
            "`mcp serve` starts a long-running MCP server and is not supported from Telegram. \
Run `zeroclaw mcp serve` on the host."
        );
    }
    anyhow::bail!("usage: mcp serve (not supported here — use host CLI)");
}

async fn dispatch_shell(config: &Config, tail: &[String]) -> Result<String> {
    if tail.first().is_some_and(|s| s == "profile") {
        let Some(name) = tail.get(1) else {
            anyhow::bail!("usage: shell profile <safe|balanced|autonomous|custom-id>");
        };
        return run_zeroclaw_cli(config, vec!["shell".into(), "profile".into(), name.clone()])
            .await;
    }
    anyhow::bail!("usage: shell profile <name>");
}

async fn dispatch_cron(config: &Config, tail: &[String]) -> Result<String> {
    match tail.first().map(|s| s.as_str()) {
        None | Some("list") => run_zeroclaw_cli(config, vec!["cron".into(), "list".into()]).await,
        Some("remove") => {
            let Some(id) = tail.get(1) else {
                anyhow::bail!("usage: cron remove <id>");
            };
            run_zeroclaw_cli(config, vec!["cron".into(), "remove".into(), id.clone()]).await
        }
        Some("add") => {
            if tail.len() < 3 {
                anyhow::bail!("usage: cron add <cron-expr> <command…>");
            }
            let expr = tail[1].clone();
            let rest = tail[2..].join(" ");
            run_zeroclaw_cli(config, vec!["cron".into(), "add".into(), expr, rest]).await
        }
        Some(other) => anyhow::bail!("unknown cron subcommand `{other}`"),
    }
}

async fn dispatch_memory(config: &Config, tail: &[String]) -> Result<String> {
    match tail.first().map(|s| s.as_str()) {
        None | Some("list") => {
            run_zeroclaw_cli(
                config,
                vec![
                    "memory".into(),
                    "list".into(),
                    "--limit".into(),
                    "50".into(),
                ],
            )
            .await
        }
        Some("stats") => run_zeroclaw_cli(config, vec!["memory".into(), "stats".into()]).await,
        Some(other) => anyhow::bail!("unknown memory subcommand `{other}`"),
    }
}

async fn dispatch_hardware(config: &Config, tail: &[String]) -> Result<String> {
    match tail.first().map(|s| s.as_str()) {
        None | Some("discover") => {
            run_zeroclaw_cli(config, vec!["hardware".into(), "discover".into()]).await
        }
        Some(other) => anyhow::bail!("unknown hardware subcommand `{other}`"),
    }
}

async fn dispatch_estop(config: &Config, tail: &[String]) -> Result<String> {
    if !config.security.estop.enabled {
        anyhow::bail!("estop is disabled in config ([security.estop].enabled = false).");
    }
    match tail.first().map(|s| s.as_str()) {
        Some("status") => run_zeroclaw_cli(config, vec!["estop".into(), "status".into()]).await,
        Some("engage") => run_zeroclaw_cli(config, vec!["estop".into(), "engage".into()]).await,
        Some("resume") => {
            anyhow::bail!(
                "`estop resume` may require an OTP or interactive confirmation. \
Use `zeroclaw estop resume` on the host."
            );
        }
        None => anyhow::bail!("usage: estop status | engage"),
        Some(other) => anyhow::bail!("unknown estop subcommand `{other}`"),
    }
}

async fn dispatch_channel(config: &Config, tail: &[String]) -> Result<String> {
    match tail.first().map(|s| s.as_str()) {
        None | Some("list") => {
            run_zeroclaw_cli(config, vec!["channel".into(), "list".into()]).await
        }
        Some("doctor") => run_zeroclaw_cli(config, vec!["channel".into(), "doctor".into()]).await,
        Some(other) => anyhow::bail!("unknown channel subcommand `{other}`"),
    }
}

async fn dispatch_config_schema(config: &Config, tail: &[String]) -> Result<String> {
    match tail.first().map(|s| s.as_str()) {
        None | Some("schema") => {
            run_zeroclaw_cli(config, vec!["config".into(), "schema".into()]).await
        }
        Some(other) => anyhow::bail!("unknown config subcommand `{other}`"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_hub_basic() {
        assert_eq!(
            parse_hub_invocation("/z skills list", "z"),
            Some(vec!["skills".to_string(), "list".to_string()])
        );
        assert_eq!(
            parse_hub_invocation("/z@mybot channel list", "z"),
            Some(vec!["channel".to_string(), "list".to_string()])
        );
        assert!(parse_hub_invocation("/x a", "z").is_none());
        assert!(parse_hub_invocation("z a", "z").is_none());
    }

    #[test]
    fn telegram_command_token_validation() {
        assert!(is_valid_telegram_command_token("z"));
        assert!(is_valid_telegram_command_token("z_tools"));
        assert!(!is_valid_telegram_command_token(""));
        assert!(!is_valid_telegram_command_token("a/b"));
    }

    #[test]
    fn sanitize_skill_name_for_menu() {
        assert_eq!(sanitize_bot_command_token("My-Skill"), "my_skill");
    }
}
