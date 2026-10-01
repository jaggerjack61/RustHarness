use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    Tokens {
        input_tokens: u64,
        output_tokens: u64,
        total_tokens: u64,
        cached_tokens: u64,
        turn_input: u64,
        turn_output: u64,
        turn_cached: u64,
        context_window: i64,
        model: String,
        reasoning_effort: Option<String>,
    },
    Thinking {
        content: String,
    },
    ThinkingDelta {
        content: String,
    },
    ThinkingEnd,
    Text {
        content: String,
    },
    TextDelta {
        content: String,
    },
    TextEnd {
        content: String,
    },
    ToolCall {
        name: String,
        arguments: Value,
    },
    ToolResult {
        name: String,
        result: String,
    },
    BackgroundToolResult {
        tool_call_id: String,
        name: String,
        arguments: Value,
        result: String,
    },
    TurnStart,
    FinishReason {
        reason: String,
        content: String,
    },
    HistoryTrimmed {
        summarized: usize,
    },
}

impl Event {
    pub fn to_value(&self) -> Value {
        serde_json::to_value(self).expect("events are always serializable")
    }

    pub fn from_value(value: Value) -> Result<Self, Value> {
        serde_json::from_value(value.clone()).map_err(|_| value)
    }
}

pub type Callback<'a> = dyn FnMut(&Event) + 'a;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_shape_matches_python_contract() {
        let event = Event::ToolCall {
            name: "read".into(),
            arguments: serde_json::json!({"path": "main.rs"}),
        };
        assert_eq!(
            event.to_value(),
            serde_json::json!({
                "type": "tool_call",
                "name": "read",
                "arguments": {"path": "main.rs"}
            })
        );
    }

    #[test]
    fn unknown_event_is_returned_unchanged() {
        let value = serde_json::json!({"type": "future_event", "value": 1});
        assert_eq!(Event::from_value(value.clone()), Err(value));
    }
}
