use std::collections::{BTreeMap, BTreeSet};

use tokenizers::Tokenizer;

use super::TextTokenizer;
use crate::error::{ModelsError, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizerValidation {
    pub vocabulary_entries: usize,
    pub max_token_id: u32,
    pub added_tokens: usize,
    pub stop_token_ids: Vec<u32>,
    pub required_token_ids: Vec<u32>,
}

impl TextTokenizer {
    pub fn validate_contract(
        &self,
        embedding_vocab_size: usize,
        required_token_ids: &[u32],
    ) -> Result<TokenizerValidation> {
        inspect(
            &self.inner,
            &self.added_tokens,
            &self.stop_token_ids(),
            self.pad_token_id,
            embedding_vocab_size,
            required_token_ids,
        )
    }
}

pub(super) fn inspect(
    tokenizer: &Tokenizer,
    added_tokens: &BTreeMap<String, u32>,
    stop_token_ids: &[u32],
    pad_token_id: Option<u32>,
    embedding_vocab_size: usize,
    required_token_ids: &[u32],
) -> Result<TokenizerValidation> {
    if embedding_vocab_size == 0 {
        return Err(invalid("embedding vocabulary is empty"));
    }
    let vocabulary = tokenizer.get_vocab(true);
    if vocabulary.is_empty() {
        return Err(invalid("tokenizer vocabulary is empty"));
    }
    let mut ids = BTreeMap::new();
    for (token, id) in &vocabulary {
        if tokenizer.token_to_id(token) != Some(*id)
            || tokenizer.id_to_token(*id).as_deref() != Some(token)
        {
            return Err(invalid(format!(
                "tokenizer mapping for {token:?} and id {id} does not round trip"
            )));
        }
        if let Some(previous) = ids.insert(*id, token)
            && previous != token
        {
            return Err(invalid(format!("token id {id} maps to multiple token contents")));
        }
    }
    let max_token_id = ids.last_key_value().map_or(0, |(id, _)| *id);
    if u64::from(max_token_id) >= u64::try_from(embedding_vocab_size)? {
        return Err(invalid(format!(
            "tokenizer id {max_token_id} exceeds embedding vocabulary size {embedding_vocab_size}"
        )));
    }
    for (token, id) in added_tokens {
        if vocabulary.get(token) != Some(id) {
            return Err(invalid(format!(
                "added token {token:?} with id {id} is absent from vocabulary"
            )));
        }
    }
    let mut required = required_token_ids.iter().copied().collect::<BTreeSet<_>>();
    required.extend(stop_token_ids.iter().copied());
    required.extend(pad_token_id);
    for id in &required {
        let token = tokenizer
            .id_to_token(*id)
            .ok_or_else(|| invalid(format!("required token id {id} is absent from tokenizer")))?;
        if tokenizer.token_to_id(&token) != Some(*id) {
            return Err(invalid(format!("required token id {id} does not round trip")));
        }
    }
    Ok(TokenizerValidation {
        vocabulary_entries: vocabulary.len(),
        max_token_id,
        added_tokens: added_tokens.len(),
        stop_token_ids: stop_token_ids.to_vec(),
        required_token_ids: required.into_iter().collect(),
    })
}

fn invalid(message: impl Into<String>) -> ModelsError {
    ModelsError::InvalidConfig(message.into())
}

#[cfg(test)]
mod tests;
