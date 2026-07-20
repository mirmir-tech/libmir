use std::{collections::BTreeMap, fs};

use serde::Deserialize;

use super::{EmbeddingTask, PoolingMode};
use crate::{
    error::{ModelsError, Result},
    layout::ModelLayout,
};

#[derive(Deserialize)]
struct Module {
    path: String,
    #[serde(rename = "type")]
    kind: String,
}

#[derive(Deserialize)]
#[expect(clippy::struct_excessive_bools, reason = "mirrors the checkpoint pooling schema")]
struct Pooling {
    word_embedding_dimension: usize,
    #[serde(rename = "pooling_mode_cls_token")]
    cls: bool,
    #[serde(rename = "pooling_mode_mean_tokens")]
    mean: bool,
    #[serde(rename = "pooling_mode_lasttoken")]
    last_token: bool,
    #[serde(default)]
    include_prompt: bool,
}

#[derive(Default, Deserialize)]
struct SentenceConfig {
    #[serde(default)]
    prompts: BTreeMap<String, String>,
    default_prompt_name: Option<String>,
}

pub(super) fn discover(layout: &ModelLayout) -> Result<Option<EmbeddingTask>> {
    let Some(modules_path) = layout.modules_path.as_ref() else {
        return Ok(None);
    };
    let modules: Vec<Module> = serde_json::from_str(&fs::read_to_string(modules_path)?)?;
    let Some(pooling_module) = modules.iter().find(|module| module.kind.ends_with(".Pooling"))
    else {
        return Ok(None);
    };
    let pooling_path = layout.root.join(&pooling_module.path).join("config.json");
    let pooling: Pooling = serde_json::from_str(&fs::read_to_string(&pooling_path)?)?;
    let mode = pooling_mode(&pooling)?;
    let normalize = modules.iter().any(|module| module.kind.ends_with(".Normalize"));
    let sentence = layout
        .sentence_transformers_config_path
        .as_ref()
        .map(|path| -> Result<SentenceConfig> {
            Ok(serde_json::from_str(&fs::read_to_string(path)?)?)
        })
        .transpose()?
        .unwrap_or_default();
    Ok(Some(EmbeddingTask {
        pooling: mode,
        normalize,
        native_dimensions: pooling.word_embedding_dimension,
        include_prompt: pooling.include_prompt,
        prompts: sentence.prompts,
        default_prompt: sentence.default_prompt_name,
    }))
}

fn pooling_mode(config: &Pooling) -> Result<PoolingMode> {
    let enabled = [
        (config.cls, PoolingMode::Cls),
        (config.mean, PoolingMode::Mean),
        (config.last_token, PoolingMode::LastToken),
    ];
    let modes: Vec<PoolingMode> = enabled
        .into_iter()
        .filter_map(|(enabled, mode)| enabled.then_some(mode))
        .collect();
    match modes.as_slice() {
        [mode] => Ok(*mode),
        [] => Err(invalid("Sentence Transformers pooling has no supported mode")),
        _ => Err(invalid("combined Sentence Transformers pooling is not yet supported")),
    }
}

fn invalid(message: impl Into<String>) -> ModelsError {
    ModelsError::InvalidConfig(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovers_exactly_one_last_token_pooling_mode() -> Result<()> {
        let pooling: Pooling = serde_json::from_value(serde_json::json!({
            "word_embedding_dimension": 1024,
            "pooling_mode_cls_token": false,
            "pooling_mode_mean_tokens": false,
            "pooling_mode_lasttoken": true,
            "include_prompt": true
        }))?;

        assert_eq!(pooling_mode(&pooling)?, PoolingMode::LastToken);
        assert!(pooling.include_prompt);
        Ok(())
    }

    #[test]
    fn rejects_combined_pooling_instead_of_guessing() {
        let pooling = Pooling {
            word_embedding_dimension: 1024,
            cls: true,
            mean: false,
            last_token: true,
            include_prompt: false,
        };
        assert!(pooling_mode(&pooling).is_err());
    }
}
