//! Single entry point for shell execution (used by the `shell` tool and skill shell).

use std::sync::Arc;
use std::time::Duration;

use crate::config::ShellSection;
use crate::runtime::RuntimeAdapter;
use crate::security::traits::Sandbox;
use crate::security::SecurityPolicy;
use crate::tools::traits::ToolResult;

use super::pipeline::{
    CommandRewriter, CommandValidator, ExtraForbiddenPathsValidator, IdentityRewriter,
    MaxLineLengthValidator, NoNulByteValidator,
};
use super::profiles::{resolve_shell_profile, ResolvedShellProfile, ShellProfileKind};

const MAX_OUTPUT_BYTES: usize = 1_048_576;
/// Sensible ceiling for a single shell invocation string (balanced/autonomous).
const MAX_COMMAND_BYTES: usize = 256 * 1024;

/// Profile-driven shell engine (immutable after construction).
pub struct ShellEngine {
    shell: ShellSection,
    resolved: ResolvedShellProfile,
    security: Arc<SecurityPolicy>,
    runtime: Arc<dyn RuntimeAdapter>,
    sandbox: Arc<dyn Sandbox>,
    validators: Vec<Arc<dyn CommandValidator>>,
    rewriters: Vec<Arc<dyn CommandRewriter>>,
    validator_count: usize,
}

impl ShellEngine {
    /// Build engine from resolved config (typically `config.shell` at startup).
    pub fn new(
        shell: ShellSection,
        security: Arc<SecurityPolicy>,
        runtime: Arc<dyn RuntimeAdapter>,
        sandbox: Arc<dyn Sandbox>,
    ) -> anyhow::Result<Self> {
        let resolved = resolve_shell_profile(&shell)?;
        let mut validators: Vec<Arc<dyn CommandValidator>> = Vec::new();
        let mut rewriters: Vec<Arc<dyn CommandRewriter>> = Vec::new();

        if !shell.safe.forbidden_paths.is_empty() {
            validators.push(Arc::new(ExtraForbiddenPathsValidator {
                paths: shell.safe.forbidden_paths.clone(),
            }));
        }

        validators.push(Arc::new(NoNulByteValidator));

        match resolved.kind {
            ShellProfileKind::Safe => {}
            ShellProfileKind::Balanced => {
                validators.push(Arc::new(MaxLineLengthValidator {
                    max_bytes: MAX_COMMAND_BYTES,
                }));
                rewriters.push(Arc::new(IdentityRewriter));
            }
            ShellProfileKind::Autonomous => {
                validators.push(Arc::new(MaxLineLengthValidator {
                    max_bytes: MAX_COMMAND_BYTES,
                }));
                rewriters.push(Arc::new(IdentityRewriter));
                #[cfg(feature = "shell-full")]
                {
                    let max_v = shell.autonomous.max_validators.max(1) as usize;
                    let mut added = 0usize;
                    if let Ok(rx) = regex::Regex::new(r"(?i)\|\s*sh\b") {
                        if added < max_v {
                            validators.push(Arc::new(
                                super::pipeline::AutonomousExtraPatternValidator { pattern: rx },
                            ));
                            added += 1;
                        }
                    }
                }
            }
        }

        let validator_count = validators.len();
        Ok(Self {
            shell,
            resolved,
            security,
            runtime,
            sandbox,
            validators,
            rewriters,
            validator_count,
        })
    }

    /// Timeout for one command (from `[shell].timeout_secs`).
    pub fn timeout_secs(&self) -> u64 {
        self.shell.timeout_secs
    }

    pub fn resolved_profile(&self) -> &ResolvedShellProfile {
        &self.resolved
    }

    fn profile_status_line(&self) -> String {
        let base = match self.resolved.kind {
            ShellProfileKind::Safe => "Safe",
            ShellProfileKind::Balanced => "Balanced",
            ShellProfileKind::Autonomous => "Autonomous",
        };
        let name = self.resolved.custom_id.as_deref().unwrap_or(base);
        format!(
            "Profile: {name} ({base}) | Validators: {}",
            self.validator_count
        )
    }

    /// Core execution path shared by the `shell` tool and skill shell tools.
    pub async fn run_command(&self, command: &str, approved: bool) -> ToolResult {
        for v in &self.validators {
            if let Err(e) = v.validate(command) {
                return ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(e),
                };
            }
        }

        if self.security.is_rate_limited() {
            return ToolResult {
                success: false,
                output: String::new(),
                error: Some("Rate limit exceeded: too many actions in the last hour".into()),
            };
        }

        match self.security.validate_command_execution(command, approved) {
            Ok(_) => {}
            Err(reason) => {
                return ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(reason),
                };
            }
        }

        if let Some(path) = self.security.forbidden_path_argument(command) {
            return ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("Path blocked by security policy: {path}")),
            };
        }

        if !self.security.record_action() {
            return ToolResult {
                success: false,
                output: String::new(),
                error: Some("Rate limit exceeded: action budget exhausted".into()),
            };
        }

        let mut cmd_str = command.to_string();
        for rw in &self.rewriters {
            cmd_str = rw.rewrite(&cmd_str);
        }

        let login = self.shell.login_shell;
        let mut cmd =
            match self
                .runtime
                .build_shell_command(&cmd_str, &self.security.workspace_dir, login)
            {
                Ok(cmd) => cmd,
                Err(e) => {
                    return ToolResult {
                        success: false,
                        output: String::new(),
                        error: Some(format!("Failed to build runtime command: {e}")),
                    };
                }
            };

        if let Err(e) = self.sandbox.wrap_command(cmd.as_std_mut()) {
            return ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("Sandbox error: {e}")),
            };
        }

        cmd.env_clear();
        for var in super::env::collect_allowed_shell_env_vars(&self.security) {
            if let Ok(val) = std::env::var(&var) {
                cmd.env(&var, val);
            }
        }

        let timeout_secs = self.shell.timeout_secs;
        let result = tokio::time::timeout(Duration::from_secs(timeout_secs), cmd.output()).await;

        let mut out = match result {
            Ok(Ok(output)) => {
                let mut stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let mut stderr = String::from_utf8_lossy(&output.stderr).to_string();

                if stdout.len() > MAX_OUTPUT_BYTES {
                    let mut b = MAX_OUTPUT_BYTES.min(stdout.len());
                    while b > 0 && !stdout.is_char_boundary(b) {
                        b -= 1;
                    }
                    stdout.truncate(b);
                    stdout.push_str("\n... [output truncated at 1MB]");
                }
                if stderr.len() > MAX_OUTPUT_BYTES {
                    let mut b = MAX_OUTPUT_BYTES.min(stderr.len());
                    while b > 0 && !stderr.is_char_boundary(b) {
                        b -= 1;
                    }
                    stderr.truncate(b);
                    stderr.push_str("\n... [stderr truncated at 1MB]");
                }

                ToolResult {
                    success: output.status.success(),
                    output: stdout,
                    error: if stderr.is_empty() {
                        None
                    } else {
                        Some(stderr)
                    },
                }
            }
            Ok(Err(e)) => ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("Failed to execute command: {e}")),
            },
            Err(_) => ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!(
                    "Command timed out after {timeout_secs}s and was killed"
                )),
            },
        };

        let header = self.profile_status_line();
        if out.success {
            out.output = format!("{header}\n{}", out.output);
        } else if let Some(ref err) = out.error {
            let e = err.clone();
            out.error = Some(format!("{header}\n{e}"));
        } else {
            out.error = Some(header);
        }
        out
    }

    /// Cron / scheduler path: validate + build `std::process::Command` (Safe-tier extras + security).
    ///
    /// Uses `[shell]` timeout and `login_shell`, but forces **Safe** validators only
    /// (plus global `[shell.safe].forbidden_paths` extras). Security policy is unchanged.
    /// Extra profile checks + host shell spawn for cron (security gates run in the scheduler).
    pub fn build_std_command_for_cron(
        shell: &ShellSection,
        command: &str,
        workspace_dir: &std::path::Path,
        security: &SecurityPolicy,
    ) -> Result<tokio::process::Command, String> {
        let v_extra = ExtraForbiddenPathsValidator {
            paths: shell.safe.forbidden_paths.clone(),
        };
        v_extra.validate(command)?;
        NoNulByteValidator.validate(command)?;

        #[cfg(not(target_os = "windows"))]
        let mut cmd = {
            let mut c = tokio::process::Command::new("sh");
            if shell.login_shell {
                c.args(["-lc", command]);
            } else {
                c.arg("-c").arg(command);
            }
            c.current_dir(workspace_dir);
            c
        };
        #[cfg(target_os = "windows")]
        let mut cmd = {
            let mut c = tokio::process::Command::new("cmd.exe");
            c.arg("/C").arg(command).current_dir(workspace_dir);
            c
        };

        cmd.stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true);

        cmd.env_clear();
        for var in super::env::collect_allowed_shell_env_vars(security) {
            if let Ok(val) = std::env::var(&var) {
                cmd.env(&var, val);
            }
        }

        Ok(cmd)
    }
}
