//! Gateway WebSocket slash commands (`/new`, `/models`, …) — shared parsing lives in
//! [`crate::channels::runtime_slash`].

use super::AppState;
use crate::channels::runtime_slash::{self, ParsedRuntimeSlash, SlashRouteSelection};
use crate::config::Config;

/// Outcome of handling a slash-only message on `/ws/chat`.
pub struct GatewaySlashResult {
    pub reply: String,
    pub clear_chat_session: bool,
    /// When set, rebuild the in-memory agent provider to match the new route.
    pub rebind: Option<(String, String, Option<String>)>,
}

fn default_gateway_route(config: &Config) -> SlashRouteSelection {
    SlashRouteSelection {
        provider: config
            .default_provider
            .clone()
            .unwrap_or_else(|| "openrouter".to_string()),
        model: config
            .default_model
            .clone()
            .unwrap_or_else(|| "anthropic/claude-sonnet-4.6".to_string()),
        api_key: None,
    }
}

fn get_gateway_route(state: &AppState, session_key: &str, config: &Config) -> SlashRouteSelection {
    state
        .gateway_chat_routes
        .lock()
        .get(session_key)
        .cloned()
        .unwrap_or_else(|| default_gateway_route(config))
}

fn set_gateway_route(
    state: &AppState,
    session_key: &str,
    next: SlashRouteSelection,
    defaults: &SlashRouteSelection,
) {
    let mut routes = state.gateway_chat_routes.lock();
    if &next == defaults {
        routes.remove(session_key);
    } else {
        routes.insert(session_key.to_string(), next);
    }
}

async fn validate_provider_init(provider_name: &str, config: &Config) -> anyhow::Result<()> {
    let pr = crate::providers::provider_runtime_options_from_config(config);
    let default_prov = config.default_provider.as_deref().unwrap_or("openrouter");
    let api_url = if provider_name == default_prov {
        config.api_url.as_deref()
    } else {
        None
    };
    let p = crate::providers::create_resilient_provider_with_options(
        provider_name,
        config.api_key.as_deref(),
        api_url,
        &config.reliability,
        &pr,
    )?;
    if let Err(e) = p.warmup().await {
        tracing::warn!(provider = %provider_name, "Provider warmup failed: {e}");
    }
    Ok(())
}

/// If `content` is a runtime slash command for gateway chat, compute the reply and side effects.
pub async fn handle_gateway_ws_slash(
    state: &AppState,
    session_key: &str,
    content: &str,
) -> Option<GatewaySlashResult> {
    if let Some(kind) = super::workspace_slash::parse_gateway_workspace_slash(content) {
        let reply = super::workspace_slash::handle_gateway_workspace_slash(state, kind).await;
        return Some(GatewaySlashResult {
            reply,
            clear_chat_session: false,
            rebind: None,
        });
    }

    let parsed = runtime_slash::parse_gateway_ws_slash(content)?;
    let config = state.config.lock().clone();
    let defaults = default_gateway_route(&config);
    let mut current = get_gateway_route(state, session_key, &config);

    match parsed {
        ParsedRuntimeSlash::NewSession => Some(GatewaySlashResult {
            reply: "Conversation history cleared. Starting fresh.".to_string(),
            clear_chat_session: true,
            rebind: None,
        }),
        ParsedRuntimeSlash::ShowProviders => Some(GatewaySlashResult {
            reply: runtime_slash::build_providers_help_response(&current),
            clear_chat_session: false,
            rebind: None,
        }),
        ParsedRuntimeSlash::ShowModel => Some(GatewaySlashResult {
            reply: runtime_slash::build_models_help_response(
                &current,
                config.workspace_dir.as_path(),
                &config.model_routes,
            ),
            clear_chat_session: false,
            rebind: None,
        }),
        ParsedRuntimeSlash::ShowConfig => Some(GatewaySlashResult {
            reply: runtime_slash::build_config_text_response(
                &current,
                config.workspace_dir.as_path(),
                &config.model_routes,
            ),
            clear_chat_session: false,
            rebind: None,
        }),
        ParsedRuntimeSlash::SetProvider(raw_provider) => {
            match runtime_slash::resolve_provider_alias(&raw_provider) {
                Some(provider_name) => {
                    match validate_provider_init(&provider_name, &config).await {
                        Ok(()) => {
                            let mut rebind = None;
                            if provider_name != current.provider {
                                current.provider.clone_from(&provider_name);
                                set_gateway_route(state, session_key, current.clone(), &defaults);
                                rebind = Some((
                                    current.provider.clone(),
                                    current.model.clone(),
                                    current.api_key.clone(),
                                ));
                            }
                            let reply = format!(
                            "Provider switched to `{provider_name}` for this browser session. Current model is `{}`.\nUse `/model <model-id>` to set a provider-compatible model.",
                            current.model
                        );
                            Some(GatewaySlashResult {
                                reply,
                                clear_chat_session: false,
                                rebind,
                            })
                        }
                        Err(err) => {
                            let safe_err = crate::providers::sanitize_api_error(&err.to_string());
                            Some(GatewaySlashResult {
                                reply: format!(
                            "Failed to initialize provider `{provider_name}`. Route unchanged.\nDetails: {safe_err}"
                        ),
                                clear_chat_session: false,
                                rebind: None,
                            })
                        }
                    }
                }
                None => Some(GatewaySlashResult {
                    reply: format!(
                        "Unknown provider `{raw_provider}`. Use `/models` to list valid providers."
                    ),
                    clear_chat_session: false,
                    rebind: None,
                }),
            }
        }
        ParsedRuntimeSlash::SetModel(raw_model) => {
            let model = raw_model.trim().trim_matches('`').to_string();
            if model.is_empty() {
                return Some(GatewaySlashResult {
                    reply: "Model ID cannot be empty. Use `/model <model-id>`.".to_string(),
                    clear_chat_session: false,
                    rebind: None,
                });
            }
            if let Some(route) = config.model_routes.iter().find(|r| {
                r.model.eq_ignore_ascii_case(&model) || r.hint.eq_ignore_ascii_case(&model)
            }) {
                current.provider = route.provider.clone();
                current.model = route.model.clone();
                current.api_key = route.api_key.clone();
            } else {
                current.model = model.clone();
            }
            set_gateway_route(state, session_key, current.clone(), &defaults);
            let reply = format!(
                "Model switched to `{}` (provider: `{}`). Context preserved.",
                current.model, current.provider
            );
            let rebind = (
                current.provider.clone(),
                current.model.clone(),
                current.api_key.clone(),
            );
            Some(GatewaySlashResult {
                reply,
                clear_chat_session: false,
                rebind: Some(rebind),
            })
        }
    }
}

/// JSON shape for [`GET /api/chat-slash-commands`](crate::gateway::api).
pub fn slash_command_catalog() -> serde_json::Value {
    serde_json::json!({
        "commands": [
            { "name": "/new", "description": "Clear this chat session and start fresh" },
            { "name": "/reset", "description": "Same as /new" },
            { "name": "/fresh-session", "description": "Same as /new (explicit fresh session)" },
            { "name": "/models", "description": "List providers or /models <provider> to switch" },
            { "name": "/model", "description": "Show models or /model <id> to switch" },
            { "name": "/config", "description": "Show current provider, model, and routes" },
            { "name": "/read", "description": "Read a workspace file from disk now (e.g. /read ok.md)" },
            { "name": "/refresh", "description": "Clear dynamic-context memo; optional path re-reads one file" },
            { "name": "/webui", "description": "Show dashboard source (/webui status) or reload from config (/webui reload)" },
        ]
    })
}
