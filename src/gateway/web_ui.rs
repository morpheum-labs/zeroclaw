//! Embedded `web/dist/` assets and optional on-disk dashboard override (`[webui]`).
//!
//! When `[webui].external_path` points at a valid Vite `dist/` directory, static files
//! are read from disk; otherwise the gateway uses compile-time embedded assets (when the
//! `embedded-web-ui` feature is enabled).

use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use axum::extract::State;
use axum::http::{header, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use parking_lot::RwLock;
use serde::Serialize;

use super::AppState;
use crate::config::Config;

/// Optional manifest inside an external `dist/` folder. When present, `schema` should be `1`.
pub const WEBUI_MANIFEST: &str = "zeroclaw-webui.json";

#[cfg(feature = "embedded-web-ui")]
#[derive(rust_embed::Embed)]
#[folder = "web/dist/"]
struct WebAssets;

#[derive(Debug, Clone, Serialize)]
pub struct WebUiStatus {
    pub source: &'static str,
    pub external_path: Option<String>,
    pub path_prefix_rewrite: bool,
}

#[derive(Clone)]
pub struct WebUiServeState {
    inner: Arc<WebUiInner>,
}

struct WebUiInner {
    active: RwLock<WebUiActive>,
}

#[derive(Clone)]
enum WebUiActive {
    Embedded,
    External {
        root_display: PathBuf,
        root_canonical: PathBuf,
    },
}

impl WebUiServeState {
    /// Build-time + runtime validation. Fails hard only when embedded assets are disabled
    /// and no usable external tree exists.
    pub fn bootstrap(config: &Config) -> anyhow::Result<Self> {
        let active = resolve_initial_active(config)?;
        log_startup(&active, config.gateway.path_prefix.as_deref().unwrap_or(""));
        Ok(Self {
            inner: Arc::new(WebUiInner {
                active: RwLock::new(active),
            }),
        })
    }

    /// Test helper: embedded-only (or external if `config.webui` is set and valid).
    pub fn for_tests(config: &Config) -> Self {
        Self::bootstrap(config).expect("web UI bootstrap in tests")
    }

    pub fn status_json(&self, path_prefix: &str) -> WebUiStatus {
        let active = self.inner.active.read().clone();
        let path_prefix_rewrite = !path_prefix.is_empty();
        match &active {
            WebUiActive::Embedded => WebUiStatus {
                source: "embedded",
                external_path: None,
                path_prefix_rewrite,
            },
            WebUiActive::External { root_display, .. } => WebUiStatus {
                source: "external",
                external_path: Some(root_display.display().to_string()),
                path_prefix_rewrite,
            },
        }
    }

    /// Re-resolve `[webui].external_path` from the given config (e.g. after `PUT /api/config`
    /// or `POST /api/webui/reload`).
    pub fn reload_from_config(&self, config: &Config) {
        let active = match try_resolve_active(config) {
            Ok(a) => a,
            Err(e) => {
                tracing::warn!("WebUI reload: {e} — keeping previous static file source");
                return;
            }
        };
        log_startup(&active, config.gateway.path_prefix.as_deref().unwrap_or(""));
        *self.inner.active.write() = active;
    }

    fn active(&self) -> WebUiActive {
        self.inner.active.read().clone()
    }

    fn downgrade_to_embedded_if_possible(&self, reason: &str) {
        #[cfg(feature = "embedded-web-ui")]
        {
            if WebAssets::get("index.html").is_some() {
                tracing::warn!(
                    "WebUI external path unusable ({reason}); falling back to embedded assets"
                );
                *self.inner.active.write() = WebUiActive::Embedded;
            }
        }
        let _ = reason;
    }
}

pub fn format_slash_status(state: &AppState) -> String {
    let s = state.web_ui.status_json(state.path_prefix.as_str());
    let mut out = format!(
        "WebUI source: {} (path-prefix rewrite: {})\n",
        s.source,
        if s.path_prefix_rewrite { "on" } else { "off" }
    );
    if let Some(p) = &s.external_path {
        out.push_str(&format!("External root: {p}\n"));
    }
    out.push_str("Reload: POST /api/webui/reload (or change config and save).");
    out
}

fn log_startup(active: &WebUiActive, path_prefix: &str) {
    let rewrite = if path_prefix.is_empty() {
        "path-prefix rewriting off"
    } else {
        "path-prefix rewriting on"
    };
    match active {
        WebUiActive::Embedded => {
            tracing::info!("WebUI source: embedded (default) ({rewrite})");
        }
        WebUiActive::External { root_display, .. } => {
            tracing::info!(
                "WebUI source: external path {} ({rewrite})",
                root_display.display()
            );
        }
    }
}

fn resolve_initial_active(config: &Config) -> anyhow::Result<WebUiActive> {
    try_resolve_active(config).map_err(|e| anyhow::anyhow!(e))
}

fn embedded_index_present() -> bool {
    #[cfg(feature = "embedded-web-ui")]
    {
        WebAssets::get("index.html").is_some()
    }
    #[cfg(not(feature = "embedded-web-ui"))]
    {
        false
    }
}

fn try_resolve_active(config: &Config) -> Result<WebUiActive, String> {
    let raw = config.webui.external_path.trim();
    if !raw.is_empty() {
        let candidate = resolve_external_path(raw, &config.workspace_dir);
        match validate_external_root(&candidate) {
            Ok((display, canonical)) => {
                return Ok(WebUiActive::External {
                    root_display: display,
                    root_canonical: canonical,
                });
            }
            Err(e) => {
                tracing::warn!(
                    "WebUI external_path {:?} is not usable ({e}) — falling back when possible",
                    raw
                );
            }
        }
    }

    #[cfg(feature = "embedded-web-ui")]
    {
        if embedded_index_present() {
            return Ok(WebUiActive::Embedded);
        }
        if raw.is_empty() {
            return Err(
                "Embedded web dashboard has no index.html (build web/dist) and [webui].external_path is unset."
                    .into(),
            );
        }
        return Err(format!(
            "Invalid [webui].external_path {:?} and embedded bundle has no index.html",
            raw
        ));
    }

    #[cfg(not(feature = "embedded-web-ui"))]
    {
        if !raw.is_empty() {
            // External was set but failed validation — error is clearer than "embedded disabled"
            return Err(
                "Invalid [webui].external_path for this binary (built without embedded web UI)."
                    .into(),
            );
        }
        Err(
            "This binary was built without embedded web assets (`embedded-web-ui` disabled). \
             Set [webui].external_path to a built `web/dist` directory (with index.html)."
                .into(),
        )
    }
}

fn resolve_external_path(raw: &str, workspace_dir: &Path) -> PathBuf {
    let expanded = shellexpand::tilde(raw.trim()).to_string();
    let p = Path::new(&expanded);
    if p.is_absolute() {
        return p.to_path_buf();
    }
    let ws = workspace_dir.join(p);
    if ws.exists() {
        return ws;
    }
    std::env::current_dir()
        .map(|cwd| cwd.join(p))
        .unwrap_or_else(|_| ws)
}

fn validate_external_root(path: &Path) -> Result<(PathBuf, PathBuf), String> {
    let meta = std::fs::metadata(path).map_err(|e| format!("{path:?}: {e}"))?;
    if !meta.is_dir() {
        return Err(format!("{path:?} is not a directory"));
    }
    let index = path.join("index.html");
    if !index.is_file() {
        return Err(format!("{path:?} has no index.html"));
    }
    let canonical = path.canonicalize().map_err(|e| format!("{path:?}: {e}"))?;
    let display = path.to_path_buf();

    let manifest = canonical.join(WEBUI_MANIFEST);
    if manifest.is_file() {
        let txt = std::fs::read_to_string(&manifest).map_err(|e| format!("{manifest:?}: {e}"))?;
        let v: serde_json::Value =
            serde_json::from_str(&txt).map_err(|e| format!("{manifest:?}: invalid JSON ({e})"))?;
        let schema = v
            .get("schema")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        if schema != 1 {
            tracing::warn!(
                "WebUI manifest {} has schema {schema} (expected 1); continuing anyway",
                manifest.display()
            );
        }
    }

    Ok((display, canonical))
}

/// Join `rel` under `root_canonical` with no `..` escape. Returns absolute file path.
fn safe_file_path(root_canonical: &Path, rel: &str) -> Option<PathBuf> {
    let rel = rel.trim_start_matches('/');
    if rel.is_empty() {
        return None;
    }
    let mut out = PathBuf::new();
    for c in Path::new(rel).components() {
        match c {
            Component::Normal(x) => out.push(x),
            Component::CurDir => {}
            Component::Prefix(_) | Component::RootDir | Component::ParentDir => return None,
        }
    }
    let full = root_canonical.join(out);
    let full_canon = full.canonicalize().ok()?;
    if full_canon.starts_with(root_canonical) {
        Some(full_canon)
    } else {
        None
    }
}

fn apply_index_transform(html: &str, path_prefix: &str) -> String {
    if path_prefix.is_empty() {
        return html.to_string();
    }
    let json_pfx = serde_json::to_string(path_prefix).unwrap_or_else(|_| "\"\"".to_string());
    let script = format!("<script>window.__ZEROCLAW_BASE__={json_pfx};</script>");
    html.replace("/_app/", &format!("{path_prefix}/_app/"))
        .replacen("<head>", &format!("<head>{script}"), 1)
}

fn response_bytes(path: &str, bytes: Vec<u8>) -> Response {
    let mime = mime_guess::from_path(path)
        .first_or_octet_stream()
        .to_string();
    let cache = if path.contains("assets/") {
        "public, max-age=31536000, immutable"
    } else {
        "no-cache"
    };
    (
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, mime),
            (header::CACHE_CONTROL, cache.to_string()),
        ],
        bytes,
    )
        .into_response()
}

fn embedded_bytes(path: &str) -> Option<Vec<u8>> {
    #[cfg(feature = "embedded-web-ui")]
    {
        WebAssets::get(path).map(|c| c.data.to_vec())
    }
    #[cfg(not(feature = "embedded-web-ui"))]
    {
        let _ = path;
        None
    }
}

fn serve_embedded_index(path_prefix: &str) -> Response {
    #[cfg(feature = "embedded-web-ui")]
    {
        let Some(content) = WebAssets::get("index.html") else {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                "Web dashboard not available. Build it with: cd web && bun install && bun run build",
            )
                .into_response();
        };
        let html = String::from_utf8_lossy(&content.data);
        let html = apply_index_transform(&html, path_prefix);
        return (
            StatusCode::OK,
            [
                (header::CONTENT_TYPE, "text/html; charset=utf-8".to_string()),
                (header::CACHE_CONTROL, "no-cache".to_string()),
            ],
            html,
        )
            .into_response();
    }
    #[cfg(not(feature = "embedded-web-ui"))]
    {
        let _ = path_prefix;
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "Web dashboard not available in this build. Set [webui].external_path to a built dist directory.",
        )
            .into_response()
    }
}

pub async fn handle_static(State(state): State<AppState>, uri: Uri) -> Response {
    let retry = state.clone();
    let path = uri
        .path()
        .strip_prefix("/_app/")
        .unwrap_or(uri.path())
        .trim_start_matches('/');

    match state.web_ui.active() {
        WebUiActive::Embedded => {
            if let Some(bytes) = embedded_bytes(path) {
                return response_bytes(path, bytes);
            }
            (StatusCode::NOT_FOUND, "Not found").into_response()
        }
        WebUiActive::External { root_canonical, .. } => {
            if !root_canonical.is_dir() || !root_canonical.join("index.html").is_file() {
                state
                    .web_ui
                    .downgrade_to_embedded_if_possible("external root missing");
                return Box::pin(handle_static(State(retry), uri)).await;
            }
            let rel = path;
            let file_path = if rel.is_empty() {
                root_canonical.join("index.html")
            } else {
                match safe_file_path(&root_canonical, rel) {
                    Some(p) => p,
                    None => return (StatusCode::NOT_FOUND, "Not found").into_response(),
                }
            };
            if file_path.is_dir() {
                return (StatusCode::NOT_FOUND, "Not found").into_response();
            }
            let mime_path = if rel.is_empty() { "index.html" } else { rel };
            if file_path.file_name().and_then(|n| n.to_str()) == Some("index.html") {
                match tokio::fs::read_to_string(&file_path).await {
                    Ok(html) => {
                        let body = apply_index_transform(&html, state.path_prefix.as_str());
                        return (
                            StatusCode::OK,
                            [
                                (header::CONTENT_TYPE, "text/html; charset=utf-8".to_string()),
                                (header::CACHE_CONTROL, "no-cache".to_string()),
                            ],
                            body,
                        )
                            .into_response();
                    }
                    Err(e) => {
                        tracing::warn!("WebUI read {}: {e}", file_path.display());
                        state
                            .web_ui
                            .downgrade_to_embedded_if_possible("index read failed");
                        return Box::pin(handle_static(State(retry), uri)).await;
                    }
                }
            }
            match tokio::fs::read(&file_path).await {
                Ok(bytes) => response_bytes(mime_path, bytes),
                Err(e) => {
                    tracing::warn!("WebUI read {}: {e}", file_path.display());
                    (StatusCode::NOT_FOUND, "Not found").into_response()
                }
            }
        }
    }
}

pub async fn handle_spa_fallback(State(state): State<AppState>) -> Response {
    let retry = state.clone();
    let path_prefix = state.path_prefix.as_str();

    match state.web_ui.active() {
        WebUiActive::Embedded => serve_embedded_index(path_prefix),
        WebUiActive::External { root_canonical, .. } => {
            let index = root_canonical.join("index.html");
            if !root_canonical.is_dir() || !index.is_file() {
                state
                    .web_ui
                    .downgrade_to_embedded_if_possible("SPA fallback: external index missing");
                return Box::pin(handle_spa_fallback(State(retry))).await;
            }
            match tokio::fs::read_to_string(&index).await {
                Ok(html) => {
                    let html = apply_index_transform(&html, path_prefix);
                    (
                        StatusCode::OK,
                        [
                            (header::CONTENT_TYPE, "text/html; charset=utf-8".to_string()),
                            (header::CACHE_CONTROL, "no-cache".to_string()),
                        ],
                        html,
                    )
                        .into_response()
                }
                Err(e) => {
                    tracing::warn!("WebUI SPA read {}: {e}", index.display());
                    state
                        .web_ui
                        .downgrade_to_embedded_if_possible("SPA read failed");
                    Box::pin(handle_spa_fallback(State(retry))).await
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safe_file_rejects_parent_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().canonicalize().unwrap();
        std::fs::write(tmp.path().join("x.txt"), "ok").unwrap();
        assert!(safe_file_path(&root, "../Cargo.toml").is_none());
        assert!(safe_file_path(&root, "x.txt").is_some());
    }

    #[test]
    fn index_transform_inserts_base() {
        let h = "<head></head><script src=\"/_app/assets/a.js\">";
        let out = apply_index_transform(h, "/zc");
        assert!(out.contains("window.__ZEROCLAW_BASE__=\"/zc\""));
        assert!(out.contains("/zc/_app/assets/a.js"));
    }
}
