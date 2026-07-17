use std::{collections::BTreeMap, fs, path::Path};

use serde_json::Value;
use tokenizers::Tokenizer;

use self::{special::SpecialTokens, token::TokenConfig};
use crate::error::{ModelsError, Result};

mod special;
mod token;

pub(super) struct TokenMetadata {
    pub(super) eos_token_ids: Vec<u32>,
}

pub(super) fn configure(
    tokenizer: &mut Tokenizer,
    tokenizer_config_path: Option<&Path>,
    added_tokens_path: Option<&Path>,
    special_tokens_map_path: Option<&Path>,
) -> Result<TokenMetadata> {
    let mut declared = configured_tokens(tokenizer_config_path)?;
    merge_legacy_tokens(&mut declared, added_tokens_path)?;
    apply_declared(tokenizer, declared)?;
    let special = special::read([tokenizer_config_path, special_tokens_map_path])?;
    apply_special_tokens(tokenizer, &special)?;
    Ok(TokenMetadata {
        eos_token_ids: eos_ids(tokenizer, special)?,
    })
}

pub(super) fn inventory(tokenizer: &Tokenizer) -> BTreeMap<String, u32> {
    tokenizer
        .get_added_tokens_decoder()
        .into_iter()
        .map(|(id, token)| (token.content, id))
        .collect()
}

fn configured_tokens(path: Option<&Path>) -> Result<BTreeMap<u32, TokenConfig>> {
    let Some(path) = path else {
        return Ok(BTreeMap::new());
    };
    let config: Value = serde_json::from_str(&fs::read_to_string(path)?)?;
    let Some(tokens) = config.get("added_tokens_decoder").and_then(Value::as_object) else {
        return Ok(BTreeMap::new());
    };
    tokens
        .iter()
        .map(|(id, token)| Ok((id.parse()?, TokenConfig::from_value(token)?)))
        .collect()
}

fn merge_legacy_tokens(
    declared: &mut BTreeMap<u32, TokenConfig>,
    path: Option<&Path>,
) -> Result<()> {
    let Some(path) = path else {
        return Ok(());
    };
    let tokens: BTreeMap<String, u32> = serde_json::from_str(&fs::read_to_string(path)?)?;
    for (content, id) in tokens {
        if let Some(configured) = declared.get(&id) {
            if configured.content != content {
                return Err(ModelsError::InvalidConfig(format!(
                    "token id {id} conflicts between tokenizer_config.json and {}",
                    path.display()
                )));
            }
            continue;
        }
        let _old = declared.insert(id, TokenConfig::legacy(content));
    }
    unique_contents(declared)
}

fn unique_contents(declared: &BTreeMap<u32, TokenConfig>) -> Result<()> {
    let mut seen = BTreeMap::new();
    for (id, token) in declared {
        if let Some(previous) = seen.insert(&token.content, id)
            && previous != id
        {
            return Err(ModelsError::InvalidConfig(format!(
                "added token {:?} has conflicting ids {previous} and {id}",
                token.content
            )));
        }
    }
    Ok(())
}

fn apply_declared(tokenizer: &mut Tokenizer, tokens: BTreeMap<u32, TokenConfig>) -> Result<()> {
    for (expected, token) in tokens {
        register(tokenizer, &token, Some(expected), "declared added token")?;
    }
    Ok(())
}

fn apply_special_tokens(tokenizer: &mut Tokenizer, special: &SpecialTokens) -> Result<()> {
    for token in &special.tokens {
        if tokenizer.token_to_id(&token.content).is_none() {
            return Err(ModelsError::InvalidConfig(format!(
                "special token {:?} has no declared tokenizer id",
                token.content,
            )));
        }
        register(tokenizer, token, None, "special token")?;
    }
    Ok(())
}

fn eos_ids(tokenizer: &Tokenizer, special: SpecialTokens) -> Result<Vec<u32>> {
    special
        .eos
        .into_iter()
        .map(|content| {
            tokenizer.token_to_id(&content).ok_or_else(|| {
                ModelsError::InvalidConfig(format!("EOS token {content:?} has no tokenizer id"))
            })
        })
        .collect()
}

fn register(
    tokenizer: &mut Tokenizer,
    token: &TokenConfig,
    expected: Option<u32>,
    source: &str,
) -> Result<()> {
    if token.content.is_empty() {
        return Err(ModelsError::InvalidConfig(format!("{source} has empty content")));
    }
    let _added = tokenizer.add_tokens(&[token.added_token()]);
    let actual = tokenizer.token_to_id(&token.content).ok_or_else(|| {
        ModelsError::InvalidConfig(format!("{source} {:?} was not added", token.content))
    })?;
    if expected.is_none_or(|expected| expected == actual) {
        return Ok(());
    }
    Err(ModelsError::InvalidConfig(format!(
        "{source} {:?} has id {}, tokenizer resolved {actual}",
        token.content,
        expected.unwrap_or_default()
    )))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use tokenizers::{
        Tokenizer,
        models::bpe::{BPE, Vocab},
    };

    use super::{apply_declared, inventory, register, token::TokenConfig};
    use crate::error::Result;

    #[test]
    fn preserves_manifest_token_ids_and_special_semantics() -> Result<()> {
        let mut vocab = Vocab::new();
        let _old = vocab.insert("base".into(), 0);
        let model = BPE::builder().vocab_and_merges(vocab, Vec::new()).build()?;
        let mut tokenizer = Tokenizer::new(model);
        let tokens = BTreeMap::from([
            (1, TokenConfig::legacy("<first>".into())),
            (2, TokenConfig::legacy("<second>".into())),
        ]);

        apply_declared(&mut tokenizer, tokens)?;
        register(
            &mut tokenizer,
            &TokenConfig::legacy("<second>".into()).special(),
            Some(2),
            "test special token",
        )?;

        assert_eq!(tokenizer.token_to_id("<first>"), Some(1));
        assert_eq!(tokenizer.token_to_id("<second>"), Some(2));
        assert_eq!(inventory(&tokenizer).get("<second>"), Some(&2));
        assert!(tokenizer.get_added_tokens_decoder()[&2].special);
        Ok(())
    }
}
