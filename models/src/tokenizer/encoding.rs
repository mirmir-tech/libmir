use super::{TextTokenizer, TokenizedPrompt};
use crate::error::Result;

pub(super) fn tokenized(encoding: &tokenizers::Encoding, bytes: usize) -> TokenizedPrompt {
    TokenizedPrompt {
        token_ids: encoding.get_ids().to_vec(),
        type_ids: encoding.get_type_ids().to_vec(),
        attention_mask: encoding.get_attention_mask().to_vec(),
        bytes,
    }
}

impl TextTokenizer {
    /// Encodes `text` and also returns the end byte offset of every token, so
    /// callers can locate text boundaries without encoding prefixes again.
    pub fn encode_with_token_ends(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> Result<(TokenizedPrompt, Vec<usize>)> {
        let encoding = self.inner.encode(text, add_special_tokens)?;
        let ends = encoding.get_offsets().iter().map(|offset| offset.1).collect();
        Ok((tokenized(&encoding, text.len()), ends))
    }
}
