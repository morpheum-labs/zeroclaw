//! Shell tool — delegates to [`crate::shell::ShellEngine`].

use super::traits::{Tool, ToolResult};
use crate::security::traits::Sandbox;
use crate::security::SecurityPolicy;
use crate::shell::ShellEngine;
use async_trait::async_trait;
use serde_json::json;
use std::sync::Arc;

/// Shell command execution tool with sandboxing (profile-driven engine).
pub struct ShellTool {
    engine: Arc<ShellEngine>,
}

impl ShellTool {
    /// Build from an existing engine (preferred when sharing with skill tools).
    pub fn from_engine(engine: Arc<ShellEngine>) -> Self {
        Self { engine }
    }

    pub fn new(
        security: Arc<SecurityPolicy>,
        runtime: Arc<dyn crate::runtime::RuntimeAdapter>,
    ) -> Self {
        let shell = crate::config::ShellSection::default();
        let sandbox = Arc::new(crate::security::NoopSandbox);
        let engine = ShellEngine::new(shell, security, runtime, sandbox)
            .expect("default shell section always resolves");
        Self {
            engine: Arc::new(engine),
        }
    }

    pub fn new_with_sandbox(
        security: Arc<SecurityPolicy>,
        runtime: Arc<dyn crate::runtime::RuntimeAdapter>,
        sandbox: Arc<dyn Sandbox>,
    ) -> Self {
        let shell = crate::config::ShellSection::default();
        let engine = ShellEngine::new(shell, security, runtime, sandbox)
            .expect("default shell section always resolves");
        Self {
            engine: Arc::new(engine),
        }
    }

    /// Shared engine handle (e.g. for skill shell tools).
    pub fn engine(&self) -> &Arc<ShellEngine> {
        &self.engine
    }
}

#[async_trait]
impl Tool for ShellTool {
    fn name(&self) -> &str {
        "shell"
    }

    fn description(&self) -> &str {
        "Execute a shell command in the workspace directory"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute"
                },
                "approved": {
                    "type": "boolean",
                    "description": "Set true to explicitly approve medium/high-risk commands in supervised mode",
                    "default": false
                }
            },
            "required": ["command"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        let command = args
            .get("command")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing 'command' parameter"))?;
        let approved = args
            .get("approved")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        Ok(self.engine.run_command(command, approved).await)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ShellSection;
    use crate::runtime::{NativeRuntime, RuntimeAdapter};
    use crate::security::{AutonomyLevel, SecurityPolicy};
    use serde_json::json;

    fn test_security(autonomy: AutonomyLevel) -> Arc<SecurityPolicy> {
        Arc::new(SecurityPolicy {
            autonomy,
            workspace_dir: std::env::temp_dir(),
            ..SecurityPolicy::default()
        })
    }

    fn test_runtime() -> Arc<dyn RuntimeAdapter> {
        Arc::new(NativeRuntime::new())
    }

    fn test_engine(autonomy: AutonomyLevel) -> Arc<ShellEngine> {
        let shell = ShellSection::default();
        Arc::new(
            ShellEngine::new(
                shell,
                test_security(autonomy),
                test_runtime(),
                Arc::new(crate::security::NoopSandbox),
            )
            .unwrap(),
        )
    }

    /// Allow `rm` through the allowlist so supervised/high-risk tests hit approval gates.
    fn test_security_rm_allowed(autonomy: AutonomyLevel) -> Arc<SecurityPolicy> {
        Arc::new(SecurityPolicy {
            autonomy,
            workspace_dir: std::env::temp_dir(),
            workspace_only: false,
            allowed_commands: vec!["*".into()],
            block_high_risk_commands: false,
            forbidden_paths: vec![],
            ..SecurityPolicy::default()
        })
    }

    fn test_engine_with_security(sec: Arc<SecurityPolicy>) -> Arc<ShellEngine> {
        Arc::new(
            ShellEngine::new(
                ShellSection::default(),
                sec,
                test_runtime(),
                Arc::new(crate::security::NoopSandbox),
            )
            .unwrap(),
        )
    }

    #[test]
    fn shell_tool_name() {
        let tool = ShellTool::from_engine(test_engine(AutonomyLevel::Supervised));
        assert_eq!(tool.name(), "shell");
    }

    #[test]
    fn shell_tool_description() {
        let tool = ShellTool::from_engine(test_engine(AutonomyLevel::Supervised));
        assert!(!tool.description().is_empty());
    }

    #[test]
    fn shell_tool_schema_has_command() {
        let tool = ShellTool::from_engine(test_engine(AutonomyLevel::Supervised));
        let schema = tool.parameters_schema();
        assert!(schema["properties"]["command"].is_object());
        assert!(schema["required"]
            .as_array()
            .expect("schema required field should be an array")
            .contains(&json!("command")));
        assert!(schema["properties"]["approved"].is_object());
    }

    #[tokio::test]
    async fn shell_executes_allowed_command() {
        let tool = ShellTool::from_engine(test_engine(AutonomyLevel::Supervised));
        let result = tool
            .execute(json!({"command": "echo hello"}))
            .await
            .expect("echo command execution should succeed");
        assert!(result.success);
        assert!(result.output.contains("hello"));
        assert!(result.output.contains("Profile:"));
        assert!(result.error.is_none());
    }

    #[tokio::test]
    async fn shell_rejects_disallowed_command() {
        let tool = ShellTool::from_engine(test_engine(AutonomyLevel::Supervised));
        let result = tool
            .execute(json!({"command": "disallowed_command_xyz"}))
            .await
            .expect("tool invocation should return structured ToolResult");
        assert!(!result.success);
        assert!(result.error.is_some());
    }

    #[tokio::test]
    async fn shell_rejects_forbidden_path() {
        let sec = Arc::new(SecurityPolicy {
            autonomy: AutonomyLevel::Supervised,
            workspace_dir: std::env::temp_dir(),
            forbidden_paths: vec!["/etc".to_string()],
            ..SecurityPolicy::default()
        });
        let shell = ShellSection::default();
        let engine = Arc::new(
            ShellEngine::new(
                shell,
                sec,
                test_runtime(),
                Arc::new(crate::security::NoopSandbox),
            )
            .unwrap(),
        );
        let tool = ShellTool::from_engine(engine);
        let result = tool
            .execute(json!({"command": "cat /etc/passwd"}))
            .await
            .expect("tool invocation should return structured ToolResult");
        assert!(!result.success);
        let err = result.error.unwrap();
        assert!(
            err.contains("forbidden") || err.contains("Path blocked"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn shell_respects_read_only_autonomy() {
        let tool = ShellTool::from_engine(test_engine(AutonomyLevel::ReadOnly));
        let result = tool
            .execute(json!({"command": "echo hello"}))
            .await
            .expect("tool invocation should return structured ToolResult");
        assert!(!result.success);
        assert!(result.error.is_some());
    }

    #[tokio::test]
    async fn shell_requires_approval_for_high_risk_in_supervised_mode() {
        let sec = test_security_rm_allowed(AutonomyLevel::Supervised);
        let marker = sec.workspace_dir.join("zeroclaw_rm_test");
        let _ = std::fs::create_dir_all(&marker);
        let tool = ShellTool::from_engine(test_engine_with_security(sec));
        let result = tool
            .execute(json!({"command": format!("rm -rf {}", marker.display())}))
            .await
            .expect("tool invocation should return structured ToolResult");
        assert!(!result.success);
        let err = result.error.unwrap();
        assert!(
            err.contains("approval") || err.contains("approved"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn shell_allows_high_risk_with_approval_in_supervised_mode() {
        let sec = test_security_rm_allowed(AutonomyLevel::Supervised);
        let marker = sec.workspace_dir.join("zeroclaw_rm_test2");
        let _ = std::fs::create_dir_all(&marker);
        let tool = ShellTool::from_engine(test_engine_with_security(sec));
        let result = tool
            .execute(json!({
                "command": format!("rm -rf {}", marker.display()),
                "approved": true
            }))
            .await
            .expect("tool invocation should return structured ToolResult");
        assert!(result.success, "{}", result.error.unwrap_or_default());
    }

    #[tokio::test]
    async fn shell_blocks_high_risk_when_blocked_by_policy() {
        let sec = Arc::new(SecurityPolicy {
            autonomy: AutonomyLevel::Full,
            workspace_dir: std::env::temp_dir(),
            allowed_commands: vec!["*".into()],
            block_high_risk_commands: true,
            ..SecurityPolicy::default()
        });
        let shell = ShellSection::default();
        let engine = Arc::new(
            ShellEngine::new(
                shell,
                sec,
                test_runtime(),
                Arc::new(crate::security::NoopSandbox),
            )
            .unwrap(),
        );
        let tool = ShellTool::from_engine(engine);
        let result = tool
            .execute(json!({"command": "rm -rf /tmp/test"}))
            .await
            .expect("tool invocation should return structured ToolResult");
        assert!(!result.success);
        let err = result.error.unwrap();
        assert!(
            err.contains("blocked") || err.contains("disallowed"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn shell_blocks_subshell_operators() {
        let tool = ShellTool::from_engine(test_engine(AutonomyLevel::Full));
        let result = tool
            .execute(json!({"command": "echo $(whoami)"}))
            .await
            .expect("tool invocation should return structured ToolResult");
        assert!(!result.success);
    }

    #[tokio::test]
    async fn shell_blocks_shell_redirections() {
        let tool = ShellTool::from_engine(test_engine(AutonomyLevel::Full));
        let result = tool
            .execute(json!({"command": "echo foo > /tmp/out.txt"}))
            .await
            .expect("tool invocation should return structured ToolResult");
        assert!(!result.success);
    }

    #[tokio::test]
    async fn shell_blocks_tee() {
        let tool = ShellTool::from_engine(test_engine(AutonomyLevel::Full));
        let result = tool
            .execute(json!({"command": "echo foo | tee /tmp/out.txt"}))
            .await
            .expect("tool invocation should return structured ToolResult");
        assert!(!result.success);
    }

    #[tokio::test]
    async fn shell_blocks_single_ampersand() {
        let tool = ShellTool::from_engine(test_engine(AutonomyLevel::Full));
        let result = tool
            .execute(json!({"command": "sleep 1 & echo hi"}))
            .await
            .expect("tool invocation should return structured ToolResult");
        assert!(!result.success);
    }

    #[tokio::test]
    async fn shell_allows_double_ampersand() {
        let tool = ShellTool::from_engine(test_engine(AutonomyLevel::Full));
        let result = tool
            .execute(json!({"command": "echo ok && echo hi"}))
            .await
            .expect("tool invocation should return structured ToolResult");
        assert!(result.success, "{}", result.error.unwrap_or_default());
    }

    #[tokio::test]
    async fn shell_blocks_dangerous_find_exec() {
        let tool = ShellTool::from_engine(test_engine(AutonomyLevel::Full));
        let result = tool
            .execute(json!({"command": "find . -exec rm {} \\;"}))
            .await
            .expect("tool invocation should return structured ToolResult");
        assert!(!result.success);
    }

    #[tokio::test]
    async fn shell_blocks_dangerous_git_config() {
        let tool = ShellTool::from_engine(test_engine(AutonomyLevel::Full));
        let result = tool
            .execute(json!({"command": "git config core.sshCommand \"rm -rf /\""}))
            .await
            .expect("tool invocation should return structured ToolResult");
        assert!(!result.success);
    }

    #[tokio::test]
    async fn shell_blocks_rate_limit() {
        let sec = Arc::new(SecurityPolicy {
            autonomy: AutonomyLevel::Full,
            workspace_dir: std::env::temp_dir(),
            max_actions_per_hour: 1,
            ..SecurityPolicy::default()
        });
        let shell = ShellSection::default();
        let engine = Arc::new(
            ShellEngine::new(
                shell,
                sec,
                test_runtime(),
                Arc::new(crate::security::NoopSandbox),
            )
            .unwrap(),
        );
        let tool = ShellTool::from_engine(engine);
        let _ = tool
            .execute(json!({"command": "echo first"}))
            .await
            .expect("first command should succeed");
        let result = tool
            .execute(json!({"command": "echo second"}))
            .await
            .expect("second command should return structured ToolResult");
        assert!(!result.success);
        let msg = result.error.unwrap();
        assert!(
            msg.contains("Rate limit exceeded"),
            "unexpected error: {msg}"
        );
    }

    #[tokio::test]
    async fn shell_blocks_rate_limit_before_execution() {
        let sec = Arc::new(SecurityPolicy {
            autonomy: AutonomyLevel::Full,
            workspace_dir: std::env::temp_dir(),
            max_actions_per_hour: 0,
            ..SecurityPolicy::default()
        });
        let shell = ShellSection::default();
        let engine = Arc::new(
            ShellEngine::new(
                shell,
                sec,
                test_runtime(),
                Arc::new(crate::security::NoopSandbox),
            )
            .unwrap(),
        );
        let tool = ShellTool::from_engine(engine);
        let result = tool
            .execute(json!({"command": "echo hi"}))
            .await
            .expect("tool invocation should return structured ToolResult");
        assert!(!result.success);
        assert!(result
            .error
            .unwrap()
            .contains("Rate limit exceeded: too many actions"));
    }

    #[tokio::test]
    async fn shell_respects_workspace_only_policy() {
        let tmp = tempfile::tempdir().unwrap();
        let workspace = tmp.path().join("ws");
        std::fs::create_dir_all(&workspace).unwrap();
        let outside = tmp.path().join("outside");
        std::fs::create_dir_all(&outside).unwrap();

        let sec = Arc::new(SecurityPolicy {
            autonomy: AutonomyLevel::Full,
            workspace_dir: workspace.clone(),
            workspace_only: true,
            ..SecurityPolicy::default()
        });
        let shell = ShellSection::default();
        let engine = Arc::new(
            ShellEngine::new(
                shell,
                sec,
                test_runtime(),
                Arc::new(crate::security::NoopSandbox),
            )
            .unwrap(),
        );
        let tool = ShellTool::from_engine(engine);

        let ok = tool
            .execute(json!({"command": format!("ls {}", workspace.display())}))
            .await
            .expect("workspace ls should succeed")
            .success;
        assert!(ok);

        let bad = tool
            .execute(json!({"command": format!("ls {}", outside.display())}))
            .await
            .expect("outside ls should return ToolResult")
            .success;
        assert!(!bad);
    }

    #[tokio::test]
    async fn shell_passthrough_shell_env_when_allowed() {
        let sec = Arc::new(SecurityPolicy {
            autonomy: AutonomyLevel::Full,
            workspace_dir: std::env::temp_dir(),
            shell_env_passthrough: vec!["ZEROCLAW_SHELL_TEST_VAR".to_string()],
            ..SecurityPolicy::default()
        });
        let shell = ShellSection::default();
        let engine = Arc::new(
            ShellEngine::new(
                shell,
                sec,
                test_runtime(),
                Arc::new(crate::security::NoopSandbox),
            )
            .unwrap(),
        );
        let tool = ShellTool::from_engine(engine);
        std::env::set_var("ZEROCLAW_SHELL_TEST_VAR", "shell-test-ok");
        let result = tool
            .execute(json!({"command": "python3 -c \"import os; print(os.environ.get('ZEROCLAW_SHELL_TEST_VAR',''))\""}))
            .await
            .expect("python should succeed");
        std::env::remove_var("ZEROCLAW_SHELL_TEST_VAR");
        assert!(result.success, "{:?}", result.error);
        assert!(result.output.contains("shell-test-ok"));
    }

    #[tokio::test]
    async fn shell_ignores_invalid_shell_env_names() {
        let sec = Arc::new(SecurityPolicy {
            autonomy: AutonomyLevel::Full,
            workspace_dir: std::env::temp_dir(),
            shell_env_passthrough: vec!["123INVALID".to_string()],
            ..SecurityPolicy::default()
        });
        let shell = ShellSection::default();
        let engine = Arc::new(
            ShellEngine::new(
                shell,
                sec,
                test_runtime(),
                Arc::new(crate::security::NoopSandbox),
            )
            .unwrap(),
        );
        let tool = ShellTool::from_engine(engine);
        let result = tool
            .execute(json!({"command": "echo hi"}))
            .await
            .expect("echo should succeed");
        assert!(result.success);
    }

    #[tokio::test]
    async fn shell_tool_can_be_constructed_with_sandbox() {
        let sec = test_security(AutonomyLevel::Supervised);
        let rt = test_runtime();
        let sandbox = Arc::new(crate::security::NoopSandbox);
        let tool = ShellTool::new_with_sandbox(sec, rt, sandbox);
        assert_eq!(tool.name(), "shell");
    }
}
