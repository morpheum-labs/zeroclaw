//! Versioned on-disk session state for Phase 5 (resume + compaction metadata).
//!
//! Interactive CLI persists to a JSON file; schema bumps allow migrations. Compaction archives
//! live under `~/.zeroclaw/sessions/archives/` as JSONL lines of [`ChatMessage`].
//!
//! ## Session scope IDs (unified naming)
//!
//! - **CLI** — SQLite memory `session_id` filter: [`memory_session_id_for_cli_path`] → `cli:<path>`.
//! - **Gateway WebSocket** — memory scope is the raw `session_id`; chat persistence uses
//!   [`gateway_backend_key`] (prefix `gw_`) so gateway sessions do not collide with channel keys.
//! - **Daemon channels** — `conversation_history_key` in `channels/mod.rs` (same string for JSONL/SQLite
//!   session backend and for memory recall).

use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::providers::ChatMessage;

/// Current on-disk schema for [`SessionRecord`].
pub const SESSION_RECORD_VERSION: u32 = 2;

/// Metadata for compaction: pointers to archived segments (full message JSONL) + latest summary hint.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct SessionCompactionMeta {
    /// Paths relative to `~/.zeroclaw/sessions/` (e.g. `archives/<uuid>.jsonl`).
    #[serde(default)]
    pub archive_paths: Vec<String>,
    /// Short excerpt of the last compaction summary (for tooling; full text remains in `history`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_summary_excerpt: Option<String>,
}

/// Persistent session record (interactive CLI and future resume paths).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionRecord {
    pub version: u32,
    pub history: Vec<ChatMessage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compaction: Option<SessionCompactionMeta>,
}

#[derive(Deserialize)]
struct LegacySessionV1 {
    #[allow(dead_code)]
    version: u32,
    history: Vec<ChatMessage>,
}

/// `~/.zeroclaw/sessions` or None if home is unavailable.
#[must_use]
pub fn sessions_root_dir() -> Option<PathBuf> {
    directories::BaseDirs::new().map(|b| b.home_dir().join(".zeroclaw").join("sessions"))
}

/// Prefix for gateway WebSocket session rows in the workspace session backend (`gw_<session_id>`).
pub const GATEWAY_SESSION_PREFIX: &str = "gw_";

/// SQLite memory + transcript scope for the interactive CLI when a session file path is set.
#[must_use]
pub fn memory_session_id_for_cli_path(path: &Path) -> Option<String> {
    let raw = path.to_string_lossy().trim().to_string();
    if raw.is_empty() {
        return None;
    }
    Some(format!("cli:{raw}"))
}

/// Key used in the workspace session store (SQLite/JSONL) for gateway chat history.
#[must_use]
pub fn gateway_backend_key(session_id: &str) -> String {
    format!("{GATEWAY_SESSION_PREFIX}{session_id}")
}

/// Delete `.jsonl` compaction archives under `archives_dir` with `modified` time older than `retention`.
pub fn gc_compaction_archives_under(
    archives_dir: &Path,
    retention: Duration,
) -> std::io::Result<usize> {
    if !archives_dir.is_dir() {
        return Ok(0);
    }

    let cutoff = SystemTime::now()
        .checked_sub(retention)
        .unwrap_or(SystemTime::UNIX_EPOCH);

    let mut removed = 0usize;
    for entry in std::fs::read_dir(archives_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
            continue;
        }
        let meta = match std::fs::metadata(&path) {
            Ok(m) => m,
            Err(_) => continue,
        };
        let modified = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
        if modified < cutoff && std::fs::remove_file(&path).is_ok() {
            removed += 1;
        }
    }
    Ok(removed)
}

/// Delete compaction archive files under `~/.zeroclaw/sessions/archives/` older than `retention`.
/// Returns the number of files removed. No-op if `sessions_root_dir()` is unavailable.
pub fn gc_compaction_archives_older_than(retention: Duration) -> std::io::Result<usize> {
    let Some(root) = sessions_root_dir() else {
        return Ok(0);
    };
    gc_compaction_archives_under(&root.join("archives"), retention)
}

fn ensure_system_prompt(history: &mut Vec<ChatMessage>, system_prompt: &str) {
    if history.is_empty() {
        history.push(ChatMessage::system(system_prompt));
    } else if history.first().map(|m| m.role.as_str()) != Some("system") {
        history.insert(0, ChatMessage::system(system_prompt));
    }
}

/// Load session file; migrate legacy v1; ensure leading system message.
pub fn load_session_record(path: &Path, system_prompt: &str) -> Result<SessionRecord> {
    if !path.exists() {
        return Ok(SessionRecord {
            version: SESSION_RECORD_VERSION,
            history: vec![ChatMessage::system(system_prompt)],
            compaction: None,
        });
    }

    let raw = std::fs::read_to_string(path).with_context(|| path.display().to_string())?;
    let value: serde_json::Value =
        serde_json::from_str(&raw).with_context(|| format!("parse {}", path.display()))?;

    let ver_u64 = value
        .get("version")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(2);
    let ver = u32::try_from(ver_u64).unwrap_or(2);

    let mut record = if ver == 1 {
        let legacy: LegacySessionV1 = serde_json::from_value(value)
            .with_context(|| format!("legacy session {}", path.display()))?;
        SessionRecord {
            version: SESSION_RECORD_VERSION,
            history: legacy.history,
            compaction: None,
        }
    } else {
        serde_json::from_value(value).with_context(|| format!("session {}", path.display()))?
    };

    record.version = SESSION_RECORD_VERSION;
    ensure_system_prompt(&mut record.history, system_prompt);
    Ok(record)
}

/// Write pretty JSON session file (v2).
pub fn save_session_record(path: &Path, record: &SessionRecord) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let payload = serde_json::to_string_pretty(record)?;
    std::fs::write(path, payload).with_context(|| path.display().to_string())?;
    Ok(())
}

/// Append one compaction segment as JSONL under `sessions/archives/`. Returns relative path from `sessions/`.
pub fn write_compaction_archive(messages: &[ChatMessage]) -> Result<Option<String>> {
    let Some(root) = sessions_root_dir() else {
        tracing::warn!("compaction archive: no home directory; skip");
        return Ok(None);
    };
    let arch = root.join("archives");
    std::fs::create_dir_all(&arch).with_context(|| arch.display().to_string())?;
    let name = format!("{}.jsonl", Uuid::new_v4());
    let full = arch.join(&name);
    let mut f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&full)
        .with_context(|| full.display().to_string())?;
    for m in messages {
        let line = serde_json::to_string(m).context("serialize ChatMessage for archive")?;
        writeln!(f, "{line}")?;
    }
    f.sync_all()?;
    Ok(Some(format!("archives/{name}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn v1_round_trips_to_v2_fields() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("session.json");
        let v1 = r#"{"version":1,"history":[{"role":"user","content":"hi"}]}"#;
        std::fs::write(&path, v1).unwrap();

        let r = load_session_record(&path, "sys").unwrap();
        assert_eq!(r.version, SESSION_RECORD_VERSION);
        assert_eq!(r.history.len(), 2);
        assert_eq!(r.history[0].role, "system");
        assert_eq!(r.history[0].content, "sys");
        assert_eq!(r.history[1].content, "hi");
        assert!(r.compaction.is_none());
    }

    #[test]
    fn save_load_round_trip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("session.json");
        let rec = SessionRecord {
            version: SESSION_RECORD_VERSION,
            history: vec![ChatMessage::system("s"), ChatMessage::user("u")],
            compaction: Some(SessionCompactionMeta {
                archive_paths: vec!["archives/x.jsonl".into()],
                last_summary_excerpt: Some("bullets".into()),
            }),
        };
        save_session_record(&path, &rec).unwrap();
        let loaded = load_session_record(&path, "fallback").unwrap();
        assert_eq!(loaded.history.len(), rec.history.len());
        assert_eq!(
            loaded.compaction.as_ref().unwrap().archive_paths,
            rec.compaction.as_ref().unwrap().archive_paths
        );
    }

    #[test]
    fn gateway_backend_key_matches_prefix() {
        assert_eq!(gateway_backend_key("abc-123"), "gw_abc-123");
    }

    #[test]
    fn gc_archives_under_removes_only_expired_jsonl() {
        use filetime::FileTime;
        use std::time::Duration;
        let dir = tempdir().unwrap();
        let arch = dir.path().join("archives");
        std::fs::create_dir_all(&arch).unwrap();
        let stale = arch.join("stale.jsonl");
        std::fs::write(&stale, b"{}\n").unwrap();
        // Make file very old (before cutoff for 1-day retention)
        let ancient = SystemTime::UNIX_EPOCH + Duration::from_secs(10_000);
        let t = FileTime::from_system_time(ancient);
        filetime::set_file_mtime(&stale, t).unwrap();

        let fresh = arch.join("fresh.jsonl");
        std::fs::write(&fresh, b"{}\n").unwrap();

        let n = gc_compaction_archives_under(&arch, Duration::from_secs(60 * 60 * 24)).unwrap();
        assert_eq!(n, 1);
        assert!(!stale.exists());
        assert!(fresh.exists());
    }
}
