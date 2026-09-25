use super::invalid;
use crate::Error;

/// A bounded excerpt of the rejected tool argument, never the conversation or
/// reasoning channel. Keep this in the typed output error rather than logging
/// whole generations. Debug escaping prevents generated control characters from
/// becoming terminal controls or additional log lines.
pub(super) fn json_error(label: &str, input: &str, error: &serde_json::Error) -> Error {
    let line = input.split('\n').nth(error.line().saturating_sub(1)).unwrap_or_default();
    let mut at = error.column().saturating_sub(1).min(line.len());
    while !line.is_char_boundary(at) {
        at -= 1;
    }
    let start = line[..at].chars().count().saturating_sub(48);
    let excerpt = line.chars().skip(start).take(96).collect::<String>();
    invalid(format!(
        "{label}: {error}; generated argument near error (untrusted): {excerpt:?}"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_context_is_bounded_unicode_safe_and_does_not_include_unrelated_prefix()
    -> crate::Result<()> {
        let input = format!("{{\"text\":\"PREFIX_SENTINEL{}\",\"bad\":NaN}}", "🛰".repeat(2000));
        let error = serde_json::from_str::<serde_json::Value>(&input)
            .err()
            .ok_or_else(|| invalid("expected invalid JSON"))?;
        let message = json_error("invalid tool JSON", &input, &error).to_string();
        assert!(message.contains("NaN"));
        assert!(message.contains("untrusted"));
        assert!(!message.contains("PREFIX_SENTINEL"));
        assert!(message.len() < 1000);
        Ok(())
    }

    #[test]
    fn context_uses_the_error_line_and_escapes_controls() -> crate::Result<()> {
        let input = "{\n\"bad\":\"\u{1b}[31m\"\n}";
        let error = serde_json::from_str::<serde_json::Value>(input)
            .err()
            .ok_or_else(|| invalid("expected invalid JSON"))?;
        let message = json_error("invalid tool JSON", input, &error).to_string();
        assert!(message.contains("bad"));
        assert!(!message.contains('\u{1b}'));
        assert!(!message.contains('\n'));
        Ok(())
    }
}
