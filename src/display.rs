use std::collections::VecDeque;
use std::sync::Mutex;

#[derive(Default)]
struct BufferState {
    complete_chunks: VecDeque<String>,
    partial_chunks: VecDeque<String>,
    complete_cache: Option<String>,
    partial_cache: Option<String>,
}

#[derive(Default)]
pub struct ResponseBuffer {
    state: Mutex<BufferState>,
}

impl ResponseBuffer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&self) {
        *self.state.lock().expect("response buffer lock poisoned") = BufferState::default();
    }

    pub fn append(&self, content: &str) {
        if content.is_empty() {
            return;
        }
        let mut state = self.state.lock().expect("response buffer lock poisoned");
        let Some(newline) = content.rfind('\n') else {
            state.partial_chunks.push_back(content.to_owned());
            state.partial_cache = None;
            return;
        };

        let mut completed: String = state.partial_chunks.drain(..).collect();
        completed.push_str(&content[..=newline]);
        state.complete_chunks.push_back(completed);
        state.complete_cache = None;

        let remainder = &content[newline + 1..];
        if remainder.is_empty() {
            state.partial_cache = Some(String::new());
        } else {
            state.partial_chunks.push_back(remainder.to_owned());
            state.partial_cache = None;
        }
    }

    pub fn set_complete(&self, text: &str) {
        let mut state = self.state.lock().expect("response buffer lock poisoned");
        state.complete_chunks = if text.is_empty() {
            VecDeque::new()
        } else {
            VecDeque::from([text.to_owned()])
        };
        state.partial_chunks.clear();
        state.complete_cache = Some(text.to_owned());
        state.partial_cache = Some(String::new());
    }

    pub fn complete(&self) -> String {
        let mut state = self.state.lock().expect("response buffer lock poisoned");
        if state.complete_cache.is_none() {
            state.complete_cache = Some(state.complete_chunks.iter().cloned().collect());
        }
        state.complete_cache.clone().unwrap_or_default()
    }

    pub fn partial(&self) -> String {
        let mut state = self.state.lock().expect("response buffer lock poisoned");
        if state.partial_cache.is_none() {
            state.partial_cache = Some(state.partial_chunks.iter().cloned().collect());
        }
        state.partial_cache.clone().unwrap_or_default()
    }

    pub fn text(&self) -> String {
        let mut state = self.state.lock().expect("response buffer lock poisoned");
        if state.complete_cache.is_none() {
            state.complete_cache = Some(state.complete_chunks.iter().cloned().collect());
        }
        if state.partial_cache.is_none() {
            state.partial_cache = Some(state.partial_chunks.iter().cloned().collect());
        }
        format!(
            "{}{}",
            state.complete_cache.as_deref().unwrap_or_default(),
            state.partial_cache.as_deref().unwrap_or_default()
        )
    }
}

#[derive(Default)]
pub struct LiveDisplay {
    pub response_buffer: ResponseBuffer,
    pub response_complete: String,
    pub streamed_any: bool,
    pub thinking_line_buf: String,
    pub thinking_first_line: bool,
    pub agent_active: bool,
}

impl LiveDisplay {
    pub fn new() -> Self {
        Self {
            thinking_first_line: true,
            ..Self::default()
        }
    }

    pub fn reset_for_new_turn(&mut self) {
        self.streamed_any = false;
        self.response_buffer.reset();
        self.response_complete.clear();
        self.thinking_line_buf.clear();
        self.thinking_first_line = true;
        self.agent_active = true;
    }

    pub fn on_event(&mut self, _event: &crate::events::Event) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn response_buffer_tracks_complete_and_partial_text() {
        let buffer = ResponseBuffer::new();
        buffer.append("first");
        buffer.append(" line\nsecond");
        assert_eq!(buffer.complete(), "first line\n");
        assert_eq!(buffer.partial(), "second");
        assert_eq!(buffer.text(), "first line\nsecond");
    }

    #[test]
    fn reset_and_set_complete_replace_state() {
        let buffer = ResponseBuffer::new();
        buffer.append("old");
        buffer.set_complete("new");
        assert_eq!(buffer.text(), "new");
        buffer.reset();
        assert_eq!(buffer.text(), "");
    }
}
