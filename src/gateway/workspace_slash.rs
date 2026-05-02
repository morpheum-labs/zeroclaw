//! Gateway WebSocket `/read` and `/refresh` — always backed by disk + cache invalidation.

use crate::config::Config;
use crate::context;
use crate::gateway::AppState;
use crate::security::SecurityPolicy;
use crate::tools::file_read::FileReadTool;
use crate::tools::Tool;
use std::sync::Arc;

/// Workspace slash variants handled before the generic runtime slash parser.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GatewayWorkspaceSlashKind {
    ReadUsage,
    ReadPath(String),
    /// Clear dynamic-context memo for this workspace (instruction fingerprint block).
    RefreshAll,
    /// Clear memo + show current bytes for one path (via `file_read` policy).
    RefreshPath(String),
    WebUiStatus,
    WebUiReload,
    WebUiUsage,
}

/// Parse `/read …` and `/refresh …` for gateway chat only.
pub fn parse_gateway_workspace_slash(content: &str) -> Option<GatewayWorkspaceSlashKind> {
    let trimmed = content.trim();
    let mut parts = trimmed.split_whitespace();
    let cmd = parts.next()?;
    let cmd_lower = cmd.to_ascii_lowercase();
    if cmd_lower == "/read" {
        let rest: Vec<&str> = parts.collect();
        if rest.is_empty() {
            return Some(GatewayWorkspaceSlashKind::ReadUsage);
        }
        return Some(GatewayWorkspaceSlashKind::ReadPath(rest.join(" ")));
    }
    if cmd_lower == "/refresh" {
        let rest: Vec<&str> = parts.collect();
        if rest.is_empty() || (rest.len() == 1 && rest[0].eq_ignore_ascii_case("all")) {
            return Some(GatewayWorkspaceSlashKind::RefreshAll);
        }
        return Some(GatewayWorkspaceSlashKind::RefreshPath(rest.join(" ")));
    }
    if cmd_lower == "/webui" {
        let sub = parts.next().unwrap_or("status");
        let sub_l = sub.to_ascii_lowercase();
        return match sub_l.as_str() {
            "status" | "help" => Some(GatewayWorkspaceSlashKind::WebUiStatus),
            "reload" => Some(GatewayWorkspaceSlashKind::WebUiReload),
            _ => Some(GatewayWorkspaceSlashKind::WebUiUsage),
        };
    }
    None
}

async fn read_workspace_path_disk(config: &Config, path: &str) -> String {
    let stamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
    let security = Arc::new(SecurityPolicy::from_config(
        &config.autonomy,
        &config.workspace_dir,
    ));
    let tool = FileReadTool::new(security);
    let args = serde_json::json!({ "path": path });
    match tool.execute(args).await {
        Ok(r) if r.success => {
            tracing::info!(%path, "gateway /read served fresh from disk");
            format!(
                "{}\n\n(read fresh from disk at {stamp})",
                r.output.trim_end()
            )
        }
        Ok(r) => {
            let detail = r
                .error
                .filter(|e| !e.is_empty())
                .unwrap_or_else(|| r.output);
            format!("{detail}\n\n(read attempted at {stamp})")
        }
        Err(e) => format!("Error: {e}\n\n(read attempted at {stamp})"),
    }
}

/// Execute a workspace slash and return plain text for the gateway reply.
pub async fn handle_gateway_workspace_slash(
    state: &AppState,
    kind: GatewayWorkspaceSlashKind,
) -> String {
    let config = state.config.lock().clone();
    match kind {
        GatewayWorkspaceSlashKind::ReadUsage => {
            "Usage: `/read <path>` — path is relative to the workspace unless policy allows otherwise.\n\
             Example: `/read ok.md` or `/read docs/README.md`"
                .to_string()
        }
        GatewayWorkspaceSlashKind::ReadPath(path) => read_workspace_path_disk(&config, &path).await,
        GatewayWorkspaceSlashKind::RefreshAll => {
            context::clear_dynamic_context_block_cache_for_workspace(config.workspace_dir.as_path());
            let stamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
            tracing::info!(
                workspace = %config.workspace_dir.display(),
                "gateway /refresh cleared dynamic-context memo for workspace"
            );
            format!(
                "Cleared dynamic-context memo for this workspace (instruction/git snapshot block). \
The next model turn will rebuild that block from disk.\n\n(refreshed at {stamp})"
            )
        }
        GatewayWorkspaceSlashKind::RefreshPath(path) => {
            context::clear_dynamic_context_block_cache_for_workspace(config.workspace_dir.as_path());
            let body = read_workspace_path_disk(&config, &path).await;
            tracing::info!(
                workspace = %config.workspace_dir.display(),
                %path,
                "gateway /refresh path cleared memo and re-read file"
            );
            format!(
                "Cleared dynamic-context memo for this workspace, then re-read `{path}` from disk.\n\n{body}"
            )
        }
        GatewayWorkspaceSlashKind::WebUiStatus => {
            super::web_ui::format_slash_status(state)
        }
        GatewayWorkspaceSlashKind::WebUiReload => {
            state.web_ui.reload_from_config(&config);
            format!(
                "WebUI configuration reloaded.\n\n{}",
                super::web_ui::format_slash_status(state)
            )
        }
        GatewayWorkspaceSlashKind::WebUiUsage => {
            "Usage: `/webui status` (default) — show active dashboard source.\n\
             `/webui reload` — re-read `[webui].external_path` from the running config."
                .to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_read_and_refresh() {
        assert_eq!(
            parse_gateway_workspace_slash("/read ok.md"),
            Some(GatewayWorkspaceSlashKind::ReadPath("ok.md".into()))
        );
        assert_eq!(
            parse_gateway_workspace_slash("  /READ  foo/bar.md  "),
            Some(GatewayWorkspaceSlashKind::ReadPath("foo/bar.md".into()))
        );
        assert_eq!(
            parse_gateway_workspace_slash("/read"),
            Some(GatewayWorkspaceSlashKind::ReadUsage)
        );
        assert_eq!(
            parse_gateway_workspace_slash("/refresh"),
            Some(GatewayWorkspaceSlashKind::RefreshAll)
        );
        assert_eq!(
            parse_gateway_workspace_slash("/refresh all"),
            Some(GatewayWorkspaceSlashKind::RefreshAll)
        );
        assert_eq!(
            parse_gateway_workspace_slash("/refresh ok.md"),
            Some(GatewayWorkspaceSlashKind::RefreshPath("ok.md".into()))
        );
        assert!(parse_gateway_workspace_slash("/new").is_none());
        assert_eq!(
            parse_gateway_workspace_slash("/webui"),
            Some(GatewayWorkspaceSlashKind::WebUiStatus)
        );
        assert_eq!(
            parse_gateway_workspace_slash("/webui status"),
            Some(GatewayWorkspaceSlashKind::WebUiStatus)
        );
        assert_eq!(
            parse_gateway_workspace_slash("/webui reload"),
            Some(GatewayWorkspaceSlashKind::WebUiReload)
        );
    }

    #[tokio::test]
    async fn read_reflects_disk_updates() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        std::fs::write(ws.join("probe.md"), "alpha").unwrap();
        let mut cfg = Config::default();
        cfg.workspace_dir = ws.to_path_buf();
        let a = super::read_workspace_path_disk(&cfg, "probe.md").await;
        assert!(a.contains("alpha"), "{a}");
        std::fs::write(ws.join("probe.md"), "beta").unwrap();
        let b = super::read_workspace_path_disk(&cfg, "probe.md").await;
        assert!(b.contains("beta"), "{b}");
    }
}
