use std::{
    fs,
    path::{Path, PathBuf},
};

use super::{AgentConfig, DecisionTensorPlan, DecisionTokenizer};
use crate::{
    error::{ModelsError, Result},
    layout::{ModernBertConfig, WeightFile},
    weights::TensorCatalog,
};

/// A Laya decision checkpoint directory:
///
/// ```text
/// model.safetensors
/// rl_agent_config.json
/// encoder/config.json
/// tokenizer/tokenizer.json
/// tokenizer/tokenizer_config.json
/// ```
#[derive(Debug, Clone)]
pub struct DecisionCheckpoint {
    pub root: PathBuf,
    pub encoder: ModernBertConfig,
    pub agent: AgentConfig,
    pub tokenizer: DecisionTokenizer,
    pub tensors: DecisionTensorPlan,
}

impl DecisionCheckpoint {
    /// Recognises a Laya checkpoint without reading its weights.
    #[must_use]
    pub fn is_decision_checkpoint(root: &Path) -> bool {
        root.join("rl_agent_config.json").is_file() && root.join("encoder/config.json").is_file()
    }

    pub fn inspect(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        let encoder = ModernBertConfig::from_path(required(root.join("encoder/config.json"))?)?;
        let agent = AgentConfig::from_path(required(root.join("rl_agent_config.json"))?)?;
        let tokenizer = DecisionTokenizer::from_files(
            required(root.join("tokenizer/tokenizer.json"))?,
            required(root.join("tokenizer/tokenizer_config.json"))?,
        )?;
        let weights = required(root.join("model.safetensors"))?;
        let weight = WeightFile {
            bytes: fs::metadata(&weights)?.len(),
            path: weights,
        };
        let catalog = TensorCatalog::from_weight_files(&[weight])?;
        let tensors = DecisionTensorPlan::discover(&encoder, agent.head_layers, &catalog)?;
        if encoder.hidden_size % 64 != 0 {
            return Err(ModelsError::InvalidConfig(
                "the decision head needs a hidden size divisible by its 64-wide heads".into(),
            ));
        }
        Ok(Self { root, encoder, agent, tokenizer, tensors })
    }

    /// Attention heads of each decision head layer.
    #[must_use]
    pub const fn head_attention_heads(&self) -> usize {
        self.encoder.hidden_size / 64
    }
}

fn required(path: PathBuf) -> Result<PathBuf> {
    if path.is_file() {
        Ok(path)
    } else {
        Err(ModelsError::MissingFile(path))
    }
}
