use super::TextTokenizer;

/// Reasoning block delimiters, opener and closer, that a vocabulary may
/// declare as single tokens.
const REASONING_BLOCKS: [(&str, &str); 2] = [("<think>", "</think>"), ("<|channel>", "<channel|>")];

impl TextTokenizer {
    /// Atomic close marker for a prompt ending inside the native think block.
    /// Other reasoning protocols and non-atomic delimiters are unsupported
    /// here.
    #[must_use]
    pub fn xml_reasoning_end_token(&self, prompt: &str) -> Option<u32> {
        if !prompt.trim_end().ends_with("<think>") {
            return None;
        }
        self.atomic("</think>")
    }

    /// Reasoning blocks the model may open itself, as (opener, closer) token
    /// pairs, when the prompt leaves the choice to it.
    #[must_use]
    pub fn reasoning_block_tokens(&self) -> Vec<(u32, u32)> {
        REASONING_BLOCKS
            .iter()
            .filter_map(|(open, close)| Some((self.atomic(open)?, self.atomic(close)?)))
            .collect()
    }

    fn atomic(&self, token: &str) -> Option<u32> {
        self.added_token_id(token).or_else(|| self.token_id(token))
    }
}
