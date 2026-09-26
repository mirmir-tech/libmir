use super::TextTokenizer;

impl TextTokenizer {
    /// Atomic close marker for a prompt ending inside the native think block.
    /// Other reasoning protocols and non-atomic delimiters are unsupported
    /// here.
    #[must_use]
    pub fn xml_reasoning_end_token(&self, prompt: &str) -> Option<u32> {
        if !prompt.trim_end().ends_with("<think>") {
            return None;
        }
        self.added_token_id("</think>").or_else(|| self.token_id("</think>"))
    }
}
