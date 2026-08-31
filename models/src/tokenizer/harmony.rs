use super::TextTokenizer;
use crate::Result;

const CHANNEL_END: &str = "<|end|>";
const TURN_START: &str = "<|start|>";
const CHANNEL: &str = "<|channel|>";
const CHANNEL_BODY: &str = "<|message|>";

impl TextTokenizer {
    /// Resolves the token sequence that closes Harmony reasoning and opens the
    /// assistant final channel. Returns `None` for non-Harmony tokenizers.
    pub fn harmony_reasoning_exit_tokens(&self) -> Result<Option<Vec<u32>>> {
        let Some(channel_end) = self.marker(CHANNEL_END) else {
            return Ok(None);
        };
        let Some(turn_start) = self.marker(TURN_START) else {
            return Ok(None);
        };
        let Some(channel) = self.marker(CHANNEL) else {
            return Ok(None);
        };
        let Some(channel_body) = self.marker(CHANNEL_BODY) else {
            return Ok(None);
        };
        let mut tokens = vec![channel_end, turn_start];
        tokens.extend(self.literal("assistant")?);
        tokens.push(channel);
        tokens.extend(self.literal("final")?);
        tokens.push(channel_body);
        Ok(Some(tokens))
    }

    fn marker(&self, token: &str) -> Option<u32> {
        self.added_token_id(token).or_else(|| self.token_id(token))
    }

    fn literal(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self.encode_with_special_tokens(text, false)?.token_ids)
    }
}
