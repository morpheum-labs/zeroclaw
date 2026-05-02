//! Pending reply routing for Telegram `ask_user` without a second `getUpdates` loop.
//!
//! Callback queries and (optional) typed replies are matched against registered waiters
//! keyed by chat and forum thread.

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

/// Telegram Bot API limit for `callback_data`.
pub const TELEGRAM_CALLBACK_DATA_MAX_BYTES: usize = 64;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct PendingChatKey {
    pub chat_id: String,
    pub thread_id: Option<String>,
}

/// Build routing key from Telegram-style `reply_target` (`chat_id` or `chat_id:topic_id`).
pub fn pending_key_from_reply_target(reply_target: &str) -> PendingChatKey {
    let reply_target = reply_target.trim();
    if let Some((chat_id, thread_id)) = reply_target.split_once(':') {
        PendingChatKey {
            chat_id: chat_id.to_string(),
            thread_id: Some(thread_id.to_string()),
        }
    } else {
        PendingChatKey {
            chat_id: reply_target.to_string(),
            thread_id: None,
        }
    }
}

enum PendingSlot {
    Choice {
        correlation: u64,
        choices: Vec<String>,
        tx: tokio::sync::oneshot::Sender<String>,
    },
    Open {
        tx: tokio::sync::oneshot::Sender<String>,
    },
}

/// Routes inline keyboard callbacks (and optional open-question text replies) to `ask_user` waiters.
pub struct TelegramPendingAskRegistry {
    /// `correlation` u64 (hex in callback payload) → chat key
    corr_index: Mutex<HashMap<u64, PendingChatKey>>,
    pending: Mutex<HashMap<PendingChatKey, PendingSlot>>,
}

impl TelegramPendingAskRegistry {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            corr_index: Mutex::new(HashMap::new()),
            pending: Mutex::new(HashMap::new()),
        })
    }

    /// Register a multiple-choice wait; returns correlation id and serialized callback stem `z{016x}` (17 chars).
    pub fn register_choice_wait(
        self: &Arc<Self>,
        key: PendingChatKey,
        choices: Vec<String>,
    ) -> anyhow::Result<(u64, tokio::sync::oneshot::Receiver<String>)> {
        anyhow::ensure!(
            !choices.is_empty(),
            "ask_user choices must be non-empty for inline keyboard"
        );
        anyhow::ensure!(
            choices.len() <= 99,
            "ask_user supports at most 99 inline choices on Telegram"
        );

        let correlation = rand_u64_nonzero();
        let (tx, rx) = tokio::sync::oneshot::channel();

        self.corr_index.lock().insert(correlation, key.clone());
        self.pending.lock().insert(
            key,
            PendingSlot::Choice {
                correlation,
                choices,
                tx,
            },
        );

        Ok((correlation, rx))
    }

    pub fn register_open_wait(
        self: &Arc<Self>,
        key: PendingChatKey,
    ) -> tokio::sync::oneshot::Receiver<String> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.pending.lock().insert(key, PendingSlot::Open { tx });
        rx
    }

    /// Remove waiter without completing (e.g. timeout).
    pub fn cancel_pending_for_chat(&self, key: &PendingChatKey) {
        let slot = self.pending.lock().remove(key);
        if let Some(PendingSlot::Choice { correlation, .. }) = slot {
            self.corr_index.lock().remove(&correlation);
        }
    }

    /// Try to satisfy a pending **choice** via callback. Returns `true` if consumed.
    pub fn complete_choice_callback(
        &self,
        chat_id: &str,
        thread_id: Option<&str>,
        callback_data: &str,
    ) -> bool {
        let Some((correlation, index)) = parse_choice_callback(callback_data) else {
            return false;
        };

        let key = {
            let guard = self.corr_index.lock();
            guard.get(&correlation).cloned()
        };

        let Some(key) = key else {
            return false;
        };
        if key.chat_id != chat_id {
            return false;
        }
        if key.thread_id.as_deref() != thread_id {
            return false;
        }

        let mut pending = self.pending.lock();
        let slot = match pending.remove(&key) {
            Some(s) => s,
            None => return false,
        };

        let PendingSlot::Choice {
            correlation: corr,
            choices,
            tx,
        } = slot
        else {
            pending.insert(key, slot);
            return false;
        };
        drop(pending);

        self.corr_index.lock().remove(&correlation);

        let answer = choices
            .get(index)
            .cloned()
            .unwrap_or_else(|| callback_data.to_string());
        let _ = tx.send(answer);
        debug_assert_eq!(corr, correlation);
        true
    }

    /// Try to satisfy a pending **open** question via a normal text message. Returns `true` if consumed.
    pub fn try_complete_open_message(
        &self,
        chat_id: &str,
        thread_id: Option<&str>,
        text: &str,
    ) -> bool {
        let key = PendingChatKey {
            chat_id: chat_id.to_string(),
            thread_id: thread_id.map(str::to_string),
        };

        let mut pending = self.pending.lock();
        let slot = match pending.remove(&key) {
            Some(s) => s,
            None => return false,
        };

        let PendingSlot::Open { tx } = slot else {
            pending.insert(key, slot);
            return false;
        };
        drop(pending);

        let _ = tx.send(text.to_string());
        true
    }

    /// If a text message arrived while a **choice** prompt is pending, deliver text as the answer.
    pub fn try_complete_choice_with_text(
        &self,
        chat_id: &str,
        thread_id: Option<&str>,
        text: &str,
    ) -> bool {
        let key = PendingChatKey {
            chat_id: chat_id.to_string(),
            thread_id: thread_id.map(str::to_string),
        };

        let mut pending = self.pending.lock();
        let slot = match pending.remove(&key) {
            Some(s) => s,
            None => return false,
        };

        let PendingSlot::Choice {
            correlation, tx, ..
        } = slot
        else {
            pending.insert(key, slot);
            return false;
        };
        drop(pending);

        self.corr_index.lock().remove(&correlation);
        let _ = tx.send(text.to_string());
        true
    }
}

fn rand_u64_nonzero() -> u64 {
    loop {
        let v = rand::random::<u64>();
        if v != 0 {
            return v;
        }
    }
}

/// Callback payload: `z` + 16 hex (`u64`) + 2-digit decimal index (00–98).
pub fn choice_callback_data(correlation: u64, choice_index: usize) -> anyhow::Result<String> {
    anyhow::ensure!(choice_index < 99, "choice index must be < 99");
    let s = format!("z{correlation:016x}{choice_index:02}");
    anyhow::ensure!(
        s.len() <= TELEGRAM_CALLBACK_DATA_MAX_BYTES,
        "callback_data exceeds Telegram limit"
    );
    Ok(s)
}

fn parse_choice_callback(data: &str) -> Option<(u64, usize)> {
    let data = data.strip_prefix('z')?;
    if data.len() < 18 {
        return None;
    }
    let (hex_part, idx_part) = data.split_at(16);
    let correlation = u64::from_str_radix(hex_part, 16).ok()?;
    if correlation == 0 {
        return None;
    }
    let index = idx_part.parse::<usize>().ok()?;
    Some((correlation, index))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn choice_callback_roundtrip() {
        let c = 0xdead_beef_cafe_f00d_u64;
        let data = choice_callback_data(c, 3).unwrap();
        assert_eq!(parse_choice_callback(&data), Some((c, 3)));
    }

    #[tokio::test]
    async fn registry_choice_callback_completes_waiter() {
        let reg = TelegramPendingAskRegistry::new();
        let key = PendingChatKey {
            chat_id: "99".into(),
            thread_id: None,
        };
        let (corr, rx) = reg
            .register_choice_wait(key.clone(), vec!["A".into(), "B".into()])
            .unwrap();
        let cb = choice_callback_data(corr, 1).unwrap();
        assert!(reg.complete_choice_callback("99", None, &cb));
        assert_eq!(rx.await.unwrap(), "B");
    }

    #[tokio::test]
    async fn registry_open_text_completes_waiter() {
        let reg = TelegramPendingAskRegistry::new();
        let key = PendingChatKey {
            chat_id: "42".into(),
            thread_id: Some("7".into()),
        };
        let rx = reg.register_open_wait(key.clone());
        assert!(reg.try_complete_open_message("42", Some("7"), "hello"));
        assert_eq!(rx.await.unwrap(), "hello");
    }
}
