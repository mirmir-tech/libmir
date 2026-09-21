use foundation::conversation::Conversation;

use super::super::ModelDescriptor;

impl ModelDescriptor {
    /// Token positions of message boundaries shared with shorter conversations.
    ///
    /// Each prefix conversation is only rendered; its boundary is located in
    /// the full prompt through the token end offsets of the single encoding.
    /// A position is a lookup key for retained state, so a boundary that a
    /// later prompt tokenizes differently costs a cache miss, never a wrong
    /// result.
    pub(in crate::model) fn cache_checkpoints(
        &self,
        conversation: &Conversation,
        full_text: &str,
        token_ends: &[usize],
        reasoning: crate::ReasoningMode,
    ) -> Vec<usize> {
        let mut checkpoints = Vec::new();
        for message_count in 1..conversation.messages.len() {
            let mut prefix = conversation.clone();
            prefix.messages.truncate(message_count);
            let prompt = self.template.render_with_reasoning(&prefix, reasoning).or_else(|_| {
                let mut boundary = conversation.messages[message_count].clone();
                boundary.content.clear();
                boundary.reasoning_content = None;
                boundary.tool_calls = None;
                prefix.messages.push(boundary);
                self.template.render_with_reasoning(&prefix, reasoning)
            });
            let Ok(prompt) = prompt else {
                continue;
            };
            let common = boundary_tokens(full_text, &prompt.text, token_ends);
            if common > 0 && common < token_ends.len() {
                checkpoints.push(common);
            }
        }
        checkpoints.sort_unstable();
        checkpoints.dedup();
        checkpoints
    }
}

/// Number of leading tokens that end inside the text shared by both renders.
fn boundary_tokens(full_text: &str, prefix_text: &str, token_ends: &[usize]) -> usize {
    let shared = full_text
        .bytes()
        .zip(prefix_text.bytes())
        .take_while(|(left, right)| left == right)
        .count();
    token_ends.iter().take_while(|end| **end <= shared).count()
}

#[cfg(test)]
mod tests {
    use super::boundary_tokens;

    #[test]
    fn boundary_counts_only_tokens_inside_the_shared_text() {
        // "<s>" carries empty offsets; "ab", "cd", "ef" end at 2, 4 and 6.
        let ends = [0, 2, 4, 6];
        assert_eq!(boundary_tokens("abcdef", "abcd", &ends), 3);
        assert_eq!(boundary_tokens("abcdef", "abc", &ends), 2);
        assert_eq!(boundary_tokens("abcdef", "abcdef", &ends), 4);
        assert_eq!(boundary_tokens("abcdef", "xbcdef", &ends), 1);
        assert_eq!(boundary_tokens("abcdef", "", &ends), 1);
    }
}
