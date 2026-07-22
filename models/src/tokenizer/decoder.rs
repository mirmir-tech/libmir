use tokenizers::{Tokenizer, tokenizer::step_decode_stream};

use super::TextTokenizer;
use crate::error::Result;

/// Stateful incremental decoder preserving whitespace and partial UTF-8 bytes.
pub struct TextDecoder<'a> {
    tokenizer: &'a Tokenizer,
    ids: Vec<u32>,
    prefix: String,
    prefix_index: usize,
}

impl TextTokenizer {
    #[must_use]
    pub fn decoder(&self) -> TextDecoder<'_> {
        TextDecoder {
            tokenizer: &self.inner,
            ids: Vec::new(),
            prefix: String::new(),
            prefix_index: 0,
        }
    }
}

impl TextDecoder<'_> {
    /// Decodes one token, returning a delta once it forms valid new text.
    pub fn step(&mut self, token_id: u32) -> Result<Option<String>> {
        Ok(step_decode_stream(
            self.tokenizer,
            vec![token_id],
            true,
            &mut self.ids,
            &mut self.prefix,
            &mut self.prefix_index,
        )?)
    }
}
