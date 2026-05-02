//! Composable shell command validators and rewriters.

/// Validates a shell command string before policy / execution.
pub trait CommandValidator: Send + Sync {
    fn validate(&self, command: &str) -> Result<(), String>;
}

/// Rewrites a command string (e.g. quoting fixes). Identity rewriter is valid.
pub trait CommandRewriter: Send + Sync {
    fn rewrite(&self, command: &str) -> String;
}

/// Extra forbidden path fragments from `[shell.safe].forbidden_paths`.
pub struct ExtraForbiddenPathsValidator {
    pub paths: Vec<String>,
}

impl CommandValidator for ExtraForbiddenPathsValidator {
    fn validate(&self, command: &str) -> Result<(), String> {
        for p in &self.paths {
            let needle = p.trim();
            if needle.is_empty() {
                continue;
            }
            if command.contains(needle) {
                return Err(format!(
                    "blocked by shell.safe.forbidden_paths (matched {needle:?})"
                ));
            }
        }
        Ok(())
    }
}

/// Reject NUL bytes which break POSIX shells unpredictably.
pub struct NoNulByteValidator;

impl CommandValidator for NoNulByteValidator {
    fn validate(&self, command: &str) -> Result<(), String> {
        if command.contains('\0') {
            return Err("command contains NUL byte".into());
        }
        Ok(())
    }
}

/// Balanced-tier: reject extremely long one-line commands (DoS guard).
pub struct MaxLineLengthValidator {
    pub max_bytes: usize,
}

impl CommandValidator for MaxLineLengthValidator {
    fn validate(&self, command: &str) -> Result<(), String> {
        if command.len() > self.max_bytes {
            return Err(format!(
                "command exceeds max length ({} > {})",
                command.len(),
                self.max_bytes
            ));
        }
        Ok(())
    }
}

/// Identity rewriter (placeholder for future balanced transforms).
pub struct IdentityRewriter;

impl CommandRewriter for IdentityRewriter {
    fn rewrite(&self, command: &str) -> String {
        command.to_string()
    }
}

#[cfg(feature = "shell-full")]
pub struct AutonomousExtraPatternValidator {
    pub pattern: regex::Regex,
}

#[cfg(feature = "shell-full")]
impl CommandValidator for AutonomousExtraPatternValidator {
    fn validate(&self, command: &str) -> Result<(), String> {
        if self.pattern.is_match(command) {
            return Err(
                "blocked by autonomous pattern policy (shell-full): matched dangerous token".into(),
            );
        }
        Ok(())
    }
}
