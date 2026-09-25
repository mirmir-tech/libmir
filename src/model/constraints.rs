use llguidance::{
    ParserFactory, token_bytes_from_tokenizer_json,
    toktrie::{TokEnv, TokTrie},
};
use toktrie_hf_tokenizers::{ByteTokenizer, ByteTokenizerEnv};

use super::ModelDescriptor;
use crate::Result;

impl ModelDescriptor {
    pub(crate) fn tool_parser_factory(&self) -> Result<&ParserFactory> {
        let cached = self.tool_parser_factory.get_or_init(|| {
            let serialized = self.tokenizer.serialized().map_err(|error| error.to_string())?;
            let env = tool_tokenizer(
                &serialized,
                self.tokenizer.vocab_size(),
                &self.tokenizer.stop_token_ids(),
            )
            .map_err(|error| error.to_string())?;
            let mut factory = ParserFactory::new_simple(&env).map_err(|error| error.to_string())?;
            factory.quiet();
            Ok(factory)
        });
        cached.as_ref().map_err(invalid)
    }
}

fn invalid(message: impl std::fmt::Display) -> crate::Error {
    models::ModelsError::InvalidConfig(format!("tool constraint tokenizer: {message}")).into()
}

pub fn tool_tokenizer(serialized: &str, vocab: usize, stops: &[u32]) -> Result<TokEnv> {
    if stops.is_empty() || stops.iter().any(|id| *id as usize >= vocab) {
        return Err(invalid("invalid stop tokens"));
    }
    let mut tokenizer = ByteTokenizer::from_json_bytes(serialized.as_bytes()).map_err(invalid)?;
    tokenizer.set_eos_tokens(stops);
    let mut env = ByteTokenizerEnv::new(tokenizer, Some(vocab)).map_err(invalid)?;
    let metadata = serde_json::from_str(serialized).map_err(invalid)?;
    let bytes = token_bytes_from_tokenizer_json(&metadata).map_err(invalid)?;
    if bytes.len() != vocab {
        return Err(invalid("token bytes vocabulary mismatch"));
    }
    // The HF adapter heuristically marks every added <...> token as special.
    // Qwen's XML tokens are ordinary text, including inside JSON strings. Honor
    // the effective tokenizer's actual `special` flags so forcing and masks agree.
    env.tok_trie = TokTrie::from(env.tok_trie.info(), &bytes).with_eos_tokens(stops);
    Ok(env.to_env())
}
