use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

use serde_json::Value;

use super::token::TokenConfig;
use crate::error::Result;

const SPECIAL_FIELDS: &[&str] =
    &["bos_token", "eos_token", "unk_token", "pad_token", "additional_special_tokens"];

pub(super) struct SpecialTokens {
    pub(super) tokens: Vec<TokenConfig>,
    pub(super) eos: Vec<String>,
}

pub(super) fn read(paths: [Option<&Path>; 2]) -> Result<SpecialTokens> {
    let mut tokens = BTreeMap::new();
    let mut eos = BTreeSet::new();
    for path in paths.into_iter().flatten() {
        let value: Value = serde_json::from_str(&fs::read_to_string(path)?)?;
        collect(&value, &mut tokens, &mut eos)?;
    }
    Ok(SpecialTokens {
        tokens: tokens.into_values().collect(),
        eos: eos.into_iter().collect(),
    })
}

fn collect(
    root: &Value,
    tokens: &mut BTreeMap<String, TokenConfig>,
    eos: &mut BTreeSet<String>,
) -> Result<()> {
    let Some(root) = root.as_object() else {
        return Ok(());
    };
    for field in SPECIAL_FIELDS {
        let Some(value) = root.get(*field) else {
            continue;
        };
        let found = collect_value(value, tokens)?;
        if *field == "eos_token" {
            eos.extend(found);
        }
    }
    Ok(())
}

fn collect_value(value: &Value, tokens: &mut BTreeMap<String, TokenConfig>) -> Result<Vec<String>> {
    match value {
        Value::String(content) => {
            let token = TokenConfig::legacy(content.clone()).special();
            let _old = tokens.insert(content.clone(), token);
            Ok(vec![content.clone()])
        },
        Value::Array(values) => values.iter().try_fold(Vec::new(), |mut found, value| {
            found.extend(collect_value(value, tokens)?);
            Ok(found)
        }),
        Value::Object(_) => {
            let token = TokenConfig::from_value(value)?.special();
            let content = token.content.clone();
            let _old = tokens.insert(content.clone(), token);
            Ok(vec![content])
        },
        Value::Null | Value::Bool(_) | Value::Number(_) => Ok(Vec::new()),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::collect;
    use crate::error::Result;

    #[test]
    fn collects_named_eos_and_additional_special_tokens() -> Result<()> {
        let value = json!({
            "eos_token": { "content": "</s>", "normalized": false },
            "additional_special_tokens": ["<image>"]
        });
        let mut tokens = std::collections::BTreeMap::new();
        let mut eos = std::collections::BTreeSet::new();

        collect(&value, &mut tokens, &mut eos)?;

        assert!(tokens.contains_key("</s>"));
        assert!(tokens.contains_key("<image>"));
        assert_eq!(eos, std::collections::BTreeSet::from(["</s>".into()]));
        Ok(())
    }
}
