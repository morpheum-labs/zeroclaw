//! [`ContextAssembler`] builds optional dynamic context for the agent loop (Phase 0–1).

use super::fingerprint::{
    compute_fingerprint, git_head_sha, instruction_files_with_mtime, ContextFingerprint,
};
use super::git_snapshot::{capture_git_snapshot, GitSnapshot};
use super::layers::collect_layered_instruction_paths;
use anyhow::Result;
use std::path::PathBuf;

/// Input paths and toggles for one assembly pass.
#[derive(Debug, Clone)]
pub struct ContextAssemblyInput {
    pub workspace: PathBuf,
    pub global_config_dir: Option<PathBuf>,
    pub user_config_dir: Option<PathBuf>,
    pub session_dir: Option<PathBuf>,
    pub options: ContextAssemblyOptions,
}

/// Toggles for dynamic context. **Default keeps behavior off** until callers opt in (Phase 1 wiring).
#[derive(Debug, Clone)]
pub struct ContextAssemblyOptions {
    /// When false, [`ContextAssembler::assemble`] returns an empty `dynamic_block` (no prompt change).
    pub enabled: bool,
    pub include_git_snapshot: bool,
    pub max_git_log_lines: usize,
}

impl Default for ContextAssemblyOptions {
    fn default() -> Self {
        Self {
            enabled: false,
            include_git_snapshot: true,
            max_git_log_lines: 5,
        }
    }
}

/// Result of assembling context; `dynamic_block` is appended to prompts when non-empty.
#[derive(Debug, Clone)]
pub struct AssembledContext {
    pub fingerprint: super::fingerprint::ContextFingerprint,
    pub dynamic_block: String,
    pub git_snapshot: Option<GitSnapshot>,
}

/// Builds hierarchical + git-aware context for the LLM.
pub trait ContextAssembler: Send + Sync {
    fn assemble(&self, input: &ContextAssemblyInput) -> Result<AssembledContext>;
}

/// Default implementation used by the runtime when Phase 1 is enabled.
#[derive(Debug, Default, Clone, Copy)]
pub struct DefaultContextAssembler;

impl DefaultContextAssembler {
    /// Instruction-file mtimes + git HEAD — **no** `git log` / snapshot work.
    #[must_use]
    pub fn fingerprint_only(self, input: &ContextAssemblyInput) -> ContextFingerprint {
        let layered = collect_layered_instruction_paths(
            input.global_config_dir.as_deref(),
            input.user_config_dir.as_deref(),
            &input.workspace,
            input.session_dir.as_deref(),
        );
        let unique_paths: Vec<PathBuf> = layered.into_iter().map(|(_, p)| p).collect();
        let with_mtime = instruction_files_with_mtime(&unique_paths);
        let head = git_head_sha(&input.workspace);
        compute_fingerprint(&with_mtime, head.as_deref())
    }
}

impl ContextAssembler for DefaultContextAssembler {
    fn assemble(&self, input: &ContextAssemblyInput) -> Result<AssembledContext> {
        let fingerprint = self.fingerprint_only(input);

        if !input.options.enabled {
            return Ok(AssembledContext {
                fingerprint,
                dynamic_block: String::new(),
                git_snapshot: None,
            });
        }

        let mut dynamic_block = String::new();
        let mut git_snapshot = None;
        if input.options.include_git_snapshot {
            if let Some(gs) =
                capture_git_snapshot(&input.workspace, input.options.max_git_log_lines)
            {
                dynamic_block.push_str(&gs.format_for_prompt());
                dynamic_block.push('\n');
                git_snapshot = Some(gs);
            }
        }

        Ok(AssembledContext {
            fingerprint,
            dynamic_block,
            git_snapshot,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_returns_empty_block_but_fingerprint() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path().to_path_buf();
        std::fs::write(ws.join("AGENTS.md"), b"x").unwrap();
        let input = ContextAssemblyInput {
            workspace: ws,
            global_config_dir: None,
            user_config_dir: None,
            session_dir: None,
            options: ContextAssemblyOptions::default(),
        };
        let asm = DefaultContextAssembler;
        let out = asm.assemble(&input).unwrap();
        assert!(out.dynamic_block.is_empty());
        assert!(out.git_snapshot.is_none());
    }

    #[test]
    fn enabled_with_git_includes_block_when_repo() {
        use std::process::Command;
        if !Command::new("git")
            .arg("--version")
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
        {
            return;
        }
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        assert!(Command::new("git")
            .args(["init"])
            .current_dir(root)
            .status()
            .unwrap()
            .success());
        std::fs::write(root.join("a.txt"), b"a").unwrap();
        assert!(Command::new("git")
            .args(["config", "user.email", "a@a.a"])
            .current_dir(root)
            .status()
            .unwrap()
            .success());
        assert!(Command::new("git")
            .args(["config", "user.name", "a"])
            .current_dir(root)
            .status()
            .unwrap()
            .success());
        assert!(Command::new("git")
            .args(["add", "a.txt"])
            .current_dir(root)
            .status()
            .unwrap()
            .success());
        assert!(Command::new("git")
            .args(["commit", "-m", "c"])
            .current_dir(root)
            .status()
            .unwrap()
            .success());

        let input = ContextAssemblyInput {
            workspace: root.to_path_buf(),
            global_config_dir: None,
            user_config_dir: None,
            session_dir: None,
            options: ContextAssemblyOptions {
                enabled: true,
                include_git_snapshot: true,
                max_git_log_lines: 3,
            },
        };
        let out = DefaultContextAssembler.assemble(&input).unwrap();
        assert!(out.dynamic_block.contains("Repository (git)"));
        assert!(out.git_snapshot.is_some());
    }

    #[test]
    fn fingerprint_changes_when_instruction_file_is_updated() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path().to_path_buf();
        let agents = ws.join("AGENTS.md");
        std::fs::write(&agents, b"v1").unwrap();
        let input_v1 = ContextAssemblyInput {
            workspace: ws.clone(),
            global_config_dir: None,
            user_config_dir: None,
            session_dir: None,
            options: ContextAssemblyOptions::default(),
        };
        let fp1 = DefaultContextAssembler.fingerprint_only(&input_v1);

        std::thread::sleep(std::time::Duration::from_millis(20));
        std::fs::write(&agents, b"v2").unwrap();
        let fp2 = DefaultContextAssembler.fingerprint_only(&input_v1);
        assert_ne!(
            fp1.0, fp2.0,
            "dynamic context fingerprint must change when layered instructions change (resume safety)"
        );
    }
}
