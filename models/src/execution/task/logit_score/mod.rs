mod tests;

use std::{
    collections::BTreeMap,
    fs,
    path::{Component, Path},
};

use serde::Deserialize;

use crate::{ModelsError, error::Result, layout::ModelLayout};

/// Binary relevance scoring from the final causal decoder logits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CausalScoringTask {
    pub true_token_id: u32,
    pub false_token_id: u32,
    pub instruction: Option<String>,
}

#[derive(Deserialize)]
struct Module {
    path: String,
    #[serde(rename = "type")]
    kind: String,
}

#[derive(Deserialize)]
struct LogitScore {
    true_token_id: u32,
    false_token_id: u32,
}

#[derive(Default, Deserialize)]
struct SentenceConfig {
    #[serde(default)]
    prompts: BTreeMap<String, String>,
    default_prompt_name: Option<String>,
}

impl CausalScoringTask {
    pub fn config_path(modules: &serde_json::Value) -> Result<Option<String>> {
        let modules: Vec<Module> = serde_json::from_value(modules.clone())?;
        let scoring: Vec<_> =
            modules.iter().filter(|module| module.kind.ends_with(".LogitScore")).collect();
        match scoring.as_slice() {
            [] => Ok(None),
            [module] => {
                if module.path.is_empty()
                    || !Path::new(&module.path)
                        .components()
                        .all(|part| matches!(part, Component::Normal(_)))
                {
                    return Err(invalid("LogitScore module path must be a relative subdirectory"));
                }
                Ok(Some(format!("{}/config.json", module.path.trim_end_matches('/'))))
            },
            _ => Err(invalid("multiple LogitScore modules are unsupported")),
        }
    }

    pub fn from_values(
        config: &serde_json::Value,
        sentence: Option<&serde_json::Value>,
        vocab_size: usize,
    ) -> Result<Self> {
        let config: LogitScore = serde_json::from_value(config.clone())?;
        if config.true_token_id == config.false_token_id
            || config.true_token_id as usize >= vocab_size
            || config.false_token_id as usize >= vocab_size
        {
            return Err(invalid(
                "LogitScore requires distinct token IDs within the decoder vocabulary",
            ));
        }
        let sentence: SentenceConfig =
            sentence.cloned().map(serde_json::from_value).transpose()?.unwrap_or_default();
        let instruction = sentence
            .default_prompt_name
            .map(|name| {
                sentence
                    .prompts
                    .get(&name)
                    .cloned()
                    .ok_or_else(|| invalid("LogitScore default prompt is missing"))
            })
            .transpose()?;
        Ok(Self {
            true_token_id: config.true_token_id,
            false_token_id: config.false_token_id,
            instruction,
        })
    }

    pub(super) fn discover(layout: &ModelLayout, vocab_size: usize) -> Result<Option<Self>> {
        let Some(path) = &layout.modules_path else {
            return Ok(None);
        };
        let modules = serde_json::from_str(&fs::read_to_string(path)?)?;
        let Some(path) = Self::config_path(&modules)? else {
            return Ok(None);
        };
        let config = serde_json::from_str(&fs::read_to_string(layout.root.join(path))?)?;
        let sentence = layout
            .sentence_transformers_config_path
            .as_ref()
            .map(|path| -> Result<serde_json::Value> {
                Ok(serde_json::from_str(&fs::read_to_string(path)?)?)
            })
            .transpose()?;
        Self::from_values(&config, sentence.as_ref(), vocab_size).map(Some)
    }
}

fn invalid(message: &str) -> ModelsError {
    ModelsError::InvalidConfig(message.into())
}
