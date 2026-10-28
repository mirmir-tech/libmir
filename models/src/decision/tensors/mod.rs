mod role;

use std::collections::BTreeMap;

pub use role::{DecisionTensor, EncoderTensor, HeadTensor, ScorerTensor};

use crate::{
    error::{ModelsError, Result},
    layout::ModernBertConfig,
    weights::{TensorCatalog, TensorInfo},
};

const ENCODER_TENSORS: [EncoderTensor; 6] = [
    EncoderTensor::AttentionNorm,
    EncoderTensor::Qkv,
    EncoderTensor::AttentionOutput,
    EncoderTensor::MlpNorm,
    EncoderTensor::MlpInput,
    EncoderTensor::MlpOutput,
];

const HEAD_TENSORS: [HeadTensor; 12] = [
    HeadTensor::AttentionNormWeight,
    HeadTensor::AttentionNormBias,
    HeadTensor::QkvWeight,
    HeadTensor::QkvBias,
    HeadTensor::AttentionOutputWeight,
    HeadTensor::AttentionOutputBias,
    HeadTensor::FeedForwardNormWeight,
    HeadTensor::FeedForwardNormBias,
    HeadTensor::UpWeight,
    HeadTensor::UpBias,
    HeadTensor::DownWeight,
    HeadTensor::DownBias,
];

const SCORER_TENSORS: [ScorerTensor; 6] = [
    ScorerTensor::NormWeight,
    ScorerTensor::NormBias,
    ScorerTensor::HiddenWeight,
    ScorerTensor::HiddenBias,
    ScorerTensor::LogitWeight,
    ScorerTensor::LogitBias,
];

/// Every inference tensor of a Laya checkpoint, checked for presence, shape,
/// and a supported floating-point encoding.
#[derive(Debug, Clone)]
pub struct DecisionTensorPlan {
    tensors: BTreeMap<DecisionTensor, TensorInfo>,
}

impl DecisionTensorPlan {
    pub fn discover(
        encoder: &ModernBertConfig,
        head_layers: usize,
        catalog: &TensorCatalog,
    ) -> Result<Self> {
        let widths = role::Widths {
            hidden: encoder.hidden_size,
            intermediate: encoder.intermediate_size,
            vocab: encoder.vocab_size,
        };
        let mut roles = vec![
            DecisionTensor::TokenEmbedding,
            DecisionTensor::EmbeddingNorm,
            DecisionTensor::FinalNorm,
            DecisionTensor::TypeEmbedding,
        ];
        for layer in 0..encoder.layers.len() {
            roles.extend(
                ENCODER_TENSORS
                    .into_iter()
                    .filter(|&tensor| layer > 0 || tensor != EncoderTensor::AttentionNorm)
                    .map(|tensor| DecisionTensor::Encoder { layer, tensor }),
            );
        }
        for layer in 0..head_layers {
            roles.extend(HEAD_TENSORS.map(|tensor| DecisionTensor::Head { layer, tensor }));
        }
        roles.extend(SCORER_TENSORS.map(DecisionTensor::Scorer));
        let tensors = roles
            .into_iter()
            .map(|role| Ok((role, checked(catalog, role, &role.shape(widths))?.clone())))
            .collect::<Result<_>>()?;
        Ok(Self { tensors })
    }

    /// The checked tensor for `role`; roles outside the plan, such as layer
    /// 0's attention norm, are an error.
    pub fn get(&self, role: DecisionTensor) -> Result<&TensorInfo> {
        self.tensors.get(&role).ok_or_else(|| {
            ModelsError::InvalidConfig(format!("{} is not part of the plan", role.name()))
        })
    }

    pub fn iter(&self) -> impl Iterator<Item = (DecisionTensor, &TensorInfo)> {
        self.tensors.iter().map(|(role, info)| (*role, info))
    }
}

fn checked<'a>(
    catalog: &'a TensorCatalog,
    role: DecisionTensor,
    shape: &[usize],
) -> Result<&'a TensorInfo> {
    let name = role.name();
    let tensor = catalog
        .get(&name)
        .ok_or_else(|| ModelsError::InvalidConfig(format!("missing Laya tensor `{name}`")))?;
    if tensor.shape != shape {
        return Err(ModelsError::InvalidConfig(format!(
            "Laya tensor `{name}` has shape {:?}, expected {shape:?}",
            tensor.shape
        )));
    }
    if !matches!(tensor.dtype.as_str(), "F16" | "BF16" | "F32") {
        return Err(ModelsError::InvalidConfig(format!(
            "Laya tensor `{name}` has unsupported dtype {}",
            tensor.dtype
        )));
    }
    Ok(tensor)
}

#[cfg(test)]
mod tests;
