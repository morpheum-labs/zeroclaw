//! Shared parsing and plain-text help for runtime slash commands (`/new`, `/models`, …).
//! Used by messaging channels and the gateway WebSocket chat.

use crate::providers;
use serde::Deserialize;
use std::fmt::Write;
use std::path::Path;

const MODEL_CACHE_FILE: &str = "models_cache.json";
const MODEL_CACHE_PREVIEW_LIMIT: usize = 10;

#[derive(Debug, Clone, Default, Deserialize)]
struct ModelCacheState {
    entries: Vec<ModelCacheEntry>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct ModelCacheEntry {
    provider: String,
    models: Vec<String>,
}

/// Provider + model selection for a sender (channel DM or gateway browser session).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlashRouteSelection {
    pub provider: String,
    pub model: String,
    pub api_key: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParsedRuntimeSlash {
    ShowProviders,
    SetProvider(String),
    ShowModel,
    SetModel(String),
    ShowConfig,
    NewSession,
}

pub fn supports_runtime_model_switch(channel_name: &str) -> bool {
    matches!(
        channel_name,
        "telegram" | "discord" | "matrix" | "slack" | "gateway"
    )
}

/// Parse `/new`, `/models`, `/model`, `/config` for `channel_name` (use `"gateway"` for web UI).
pub fn parse_runtime_slash(channel_name: &str, content: &str) -> Option<ParsedRuntimeSlash> {
    let trimmed = content.trim();
    if !trimmed.starts_with('/') {
        return None;
    }

    let mut parts = trimmed.split_whitespace();
    let command_token = parts.next()?;
    let base_command = command_token
        .split('@')
        .next()
        .unwrap_or(command_token)
        .to_ascii_lowercase();

    match base_command.as_str() {
        "/new" => Some(ParsedRuntimeSlash::NewSession),
        "/models" if supports_runtime_model_switch(channel_name) => {
            if let Some(provider) = parts.next() {
                Some(ParsedRuntimeSlash::SetProvider(provider.trim().to_string()))
            } else {
                Some(ParsedRuntimeSlash::ShowProviders)
            }
        }
        "/model" if supports_runtime_model_switch(channel_name) => {
            let model = parts.collect::<Vec<_>>().join(" ").trim().to_string();
            if model.is_empty() {
                Some(ParsedRuntimeSlash::ShowModel)
            } else {
                Some(ParsedRuntimeSlash::SetModel(model))
            }
        }
        "/config" if supports_runtime_model_switch(channel_name) => {
            Some(ParsedRuntimeSlash::ShowConfig)
        }
        _ => None,
    }
}

/// WebSocket chat: treat `/reset` like `/new` (OpenClaw-style), then delegate to [`parse_runtime_slash`].
pub fn parse_gateway_ws_slash(content: &str) -> Option<ParsedRuntimeSlash> {
    let trimmed = content.trim();
    let base = trimmed
        .split_whitespace()
        .next()
        .unwrap_or("")
        .split('@')
        .next()
        .unwrap_or("")
        .to_ascii_lowercase();
    if base == "/reset" || base == "/fresh-session" {
        return Some(ParsedRuntimeSlash::NewSession);
    }
    parse_runtime_slash("gateway", content)
}

pub fn resolve_provider_alias(name: &str) -> Option<String> {
    let candidate = name.trim();
    if candidate.is_empty() {
        return None;
    }

    let providers_list = providers::list_providers();
    for provider in providers_list {
        if provider.name.eq_ignore_ascii_case(candidate)
            || provider
                .aliases
                .iter()
                .any(|alias| alias.eq_ignore_ascii_case(candidate))
        {
            return Some(provider.name.to_string());
        }
    }

    None
}

pub fn load_cached_model_preview(workspace_dir: &Path, provider_name: &str) -> Vec<String> {
    let cache_path = workspace_dir.join("state").join(MODEL_CACHE_FILE);
    let Ok(raw) = std::fs::read_to_string(cache_path) else {
        return Vec::new();
    };
    let Ok(state) = serde_json::from_str::<ModelCacheState>(&raw) else {
        return Vec::new();
    };

    state
        .entries
        .into_iter()
        .find(|entry| entry.provider == provider_name)
        .map(|entry| {
            entry
                .models
                .into_iter()
                .take(MODEL_CACHE_PREVIEW_LIMIT)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

pub fn build_models_help_response(
    current: &SlashRouteSelection,
    workspace_dir: &Path,
    model_routes: &[crate::config::ModelRouteConfig],
) -> String {
    let mut response = String::new();
    let _ = writeln!(
        response,
        "Current provider: `{}`\nCurrent model: `{}`",
        current.provider, current.model
    );
    response.push_str("\nSwitch model with `/model <model-id>` or `/model <hint>`.\n");

    if !model_routes.is_empty() {
        response.push_str("\nConfigured model routes:\n");
        for route in model_routes {
            let _ = writeln!(
                response,
                "  `{}` → {} ({})",
                route.hint, route.model, route.provider
            );
        }
    }

    let cached_models = load_cached_model_preview(workspace_dir, &current.provider);
    if cached_models.is_empty() {
        let _ = writeln!(
            response,
            "\nNo cached model list found for `{}`. Ask the operator to run `zeroclaw models refresh --provider {}`.",
            current.provider, current.provider
        );
    } else {
        let _ = writeln!(
            response,
            "\nCached model IDs (top {}):",
            cached_models.len()
        );
        for model in cached_models {
            let _ = writeln!(response, "- `{model}`");
        }
    }

    response
}

pub fn build_providers_help_response(current: &SlashRouteSelection) -> String {
    let mut response = String::new();
    let _ = writeln!(
        response,
        "Current provider: `{}`\nCurrent model: `{}`",
        current.provider, current.model
    );
    response.push_str("\nSwitch provider with `/models <provider>`.\n");
    response.push_str("Switch model with `/model <model-id>`.\n\n");
    response.push_str("Available providers:\n");
    for provider in providers::list_providers() {
        if provider.aliases.is_empty() {
            let _ = writeln!(response, "- {}", provider.name);
        } else {
            let _ = writeln!(
                response,
                "- {} (aliases: {})",
                provider.name,
                provider.aliases.join(", ")
            );
        }
    }
    response
}

/// Plain-text `/config` response (non-Slack).
pub fn build_config_text_response(
    current: &SlashRouteSelection,
    _workspace_dir: &Path,
    model_routes: &[crate::config::ModelRouteConfig],
) -> String {
    let mut resp = String::new();
    let _ = writeln!(
        resp,
        "Current provider: `{}`\nCurrent model: `{}`",
        current.provider, current.model
    );
    resp.push_str("\nAvailable providers:\n");
    for p in providers::list_providers() {
        let _ = writeln!(resp, "- `{}`", p.name);
    }
    if !model_routes.is_empty() {
        resp.push_str("\nConfigured model routes:\n");
        for route in model_routes {
            let _ = writeln!(
                resp,
                "  `{}` -> {} ({})",
                route.hint, route.model, route.provider
            );
        }
    }
    resp.push_str(
        "\nUse `/models <provider>` to switch provider.\nUse `/model <model-id>` to switch model.",
    );
    resp
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gateway_supports_model_slash() {
        assert!(supports_runtime_model_switch("gateway"));
        assert!(parse_runtime_slash("gateway", "/models").is_some());
        assert!(parse_runtime_slash("gateway", "/model").is_some());
        assert!(parse_runtime_slash("gateway", "/config").is_some());
        assert!(parse_runtime_slash("imessage", "/models").is_none());
    }

    #[test]
    fn parse_reset_as_new_session() {
        assert_eq!(
            parse_gateway_ws_slash("/reset"),
            Some(ParsedRuntimeSlash::NewSession)
        );
        assert_eq!(
            parse_gateway_ws_slash("  /RESET  "),
            Some(ParsedRuntimeSlash::NewSession)
        );
        assert_eq!(
            parse_gateway_ws_slash("/fresh-session"),
            Some(ParsedRuntimeSlash::NewSession)
        );
    }

    #[test]
    fn parse_new_session() {
        assert_eq!(
            parse_runtime_slash("gateway", "/new"),
            Some(ParsedRuntimeSlash::NewSession)
        );
    }
}
