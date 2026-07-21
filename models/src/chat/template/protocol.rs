use std::{fs, path::Path};

use serde_json::Value;

use crate::error::Result;

const CHATML_START: &str = "<|im_start|>";
const CHATML_END: &str = "<|im_end|>";

pub(super) fn has_chatml_tokens(path: Option<&Path>) -> Result<bool> {
    let Some(path) = path.filter(|path| path.extension().is_some_and(|ext| ext == "json")) else {
        return Ok(false);
    };
    let tokenizer: Value = serde_json::from_str(&fs::read_to_string(path)?)?;
    Ok(has_added_token(&tokenizer, CHATML_START) && has_added_token(&tokenizer, CHATML_END))
}

fn has_added_token(tokenizer: &Value, expected: &str) -> bool {
    tokenizer.get("added_tokens").and_then(Value::as_array).is_some_and(|tokens| {
        tokens
            .iter()
            .any(|token| token.get("content").and_then(Value::as_str) == Some(expected))
    })
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn recognizes_complete_chatml_protocol() {
        let tokenizer = json!({
            "added_tokens": [
                { "content": CHATML_START, "special": true },
                { "content": CHATML_END, "special": true }
            ]
        });

        assert!(has_added_token(&tokenizer, CHATML_START));
        assert!(has_added_token(&tokenizer, CHATML_END));
    }

    #[test]
    fn rejects_partial_chatml_protocol() {
        let tokenizer = json!({
            "added_tokens": [{ "content": CHATML_END, "special": true }]
        });

        assert!(!has_added_token(&tokenizer, CHATML_START));
        assert!(has_added_token(&tokenizer, CHATML_END));
    }
}
