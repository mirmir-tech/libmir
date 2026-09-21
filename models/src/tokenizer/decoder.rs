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

    /// Returns text withheld as a partial UTF-8 sequence once generation ends,
    /// decoded as the tokenizer decodes the complete sequence.
    pub fn finish(&mut self) -> Result<Option<String>> {
        if self.ids.is_empty() {
            return Ok(None);
        }
        let text = self.tokenizer.decode(&self.ids, true)?;
        let rest = text.get(self.prefix.len()..).unwrap_or_default().to_owned();
        self.ids.clear();
        self.prefix.clear();
        self.prefix_index = 0;
        Ok((!rest.is_empty()).then_some(rest))
    }
}

#[cfg(test)]
mod tests {
    use tokenizers::{
        Tokenizer,
        decoders::byte_level::ByteLevel,
        models::bpe::{BPE, Vocab},
    };

    use super::TextDecoder;

    // Byte-level spellings of the two UTF-8 bytes of U+00E9.
    fn tokenizer() -> Result<Tokenizer, Box<dyn std::error::Error + Send + Sync>> {
        let vocab = Vocab::from_iter([
            ("\u{c3}".to_owned(), 0),
            ("\u{a9}".to_owned(), 1),
            ("a".to_owned(), 2),
        ]);
        let mut tokenizer =
            Tokenizer::new(BPE::builder().vocab_and_merges(vocab, Vec::new()).build()?);
        tokenizer.with_decoder(Some(ByteLevel::default()));
        Ok(tokenizer)
    }

    fn decoder(tokenizer: &Tokenizer) -> TextDecoder<'_> {
        TextDecoder {
            tokenizer,
            ids: Vec::new(),
            prefix: String::new(),
            prefix_index: 0,
        }
    }

    #[test]
    fn finish_releases_a_trailing_partial_sequence_once()
    -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let tokenizer = tokenizer()?;
        let mut decoder = decoder(&tokenizer);
        assert_eq!(decoder.step(2)?, Some("a".into()));
        assert_eq!(decoder.step(0)?, None);
        assert_eq!(decoder.finish()?, Some("\u{fffd}".into()));
        assert_eq!(decoder.finish()?, None);
        Ok(())
    }

    #[test]
    fn finish_is_empty_after_complete_text() -> Result<(), Box<dyn std::error::Error + Send + Sync>>
    {
        let tokenizer = tokenizer()?;
        let mut decoder = decoder(&tokenizer);
        assert_eq!(decoder.step(0)?, None);
        assert_eq!(decoder.step(1)?, Some("\u{e9}".into()));
        assert_eq!(decoder.finish()?, None);
        Ok(())
    }
}
