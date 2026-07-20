use crate::{
    error::{ModelsError, Result},
    layout::DecoderConfig,
    weights::{DecoderTensorSchema, TensorCatalog},
};

mod task;

pub use task::{EmbeddingTask, ModelTask, PoolingMode, SequenceScoringTask, TaskExecutionPlan};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecoderArchetype {
    HybridMoe,
    HybridLinearMoe,
    DenseSwiGlu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionFeature {
    RmsNormalizedSharedKv,
    GatedDeltaAndRmsNormalizedGroupedQuery,
    RmsNormalizedGroupedQuery,
    GroupedQuery,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeedForwardFeature {
    DenseGeluAndRoutedMoe,
    SharedExpertRoutedSwiGlu,
    DenseSwiGlu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutionPlan {
    pub decoder: DecoderArchetype,
    pub attention: AttentionFeature,
    pub feed_forward: FeedForwardFeature,
}

impl ExecutionPlan {
    pub fn discover(decoder: &DecoderConfig, tensors: &TensorCatalog) -> Result<Self> {
        let decoder_ready =
            DecoderTensorSchema::discover(decoder, tensors).readiness(tensors).is_ready();
        if hybrid_linear_moe_layout(tensors)
            || (decoder_ready && decoder.uses_hybrid_linear_moe_stack())
        {
            if !decoder.uses_hybrid_linear_moe_stack() {
                return Err(invalid(
                    "hybrid linear MoE tensor layout has incompatible decoder features",
                ));
            }
            return Ok(Self {
                decoder: DecoderArchetype::HybridLinearMoe,
                attention: AttentionFeature::GatedDeltaAndRmsNormalizedGroupedQuery,
                feed_forward: FeedForwardFeature::SharedExpertRoutedSwiGlu,
            });
        }
        if has_all(tensors, HYBRID_MOE_LAYOUT)
            || (decoder_ready && decoder.uses_hybrid_routed_moe_stack())
        {
            if !decoder.uses_hybrid_routed_moe_stack() {
                return Err(invalid("hybrid MoE tensor layout has incompatible decoder features"));
            }
            return Ok(Self {
                decoder: DecoderArchetype::HybridMoe,
                attention: AttentionFeature::RmsNormalizedSharedKv,
                feed_forward: FeedForwardFeature::DenseGeluAndRoutedMoe,
            });
        }
        if dense_swiglu_layout(decoder, tensors) || (decoder_ready && decoder.num_experts.is_none())
        {
            if decoder.num_experts.is_some() {
                return Err(invalid("dense SwiGLU tensor layout cannot satisfy MoE features"));
            }
            return Ok(Self {
                decoder: DecoderArchetype::DenseSwiGlu,
                attention: dense_attention_feature(tensors),
                feed_forward: FeedForwardFeature::DenseSwiGlu,
            });
        }
        Err(invalid("no supported decoder tensor layout"))
    }

    #[must_use]
    pub const fn is_native_implemented(self) -> bool {
        true
    }
}

fn has_all(tensors: &TensorCatalog, names: &[&str]) -> bool {
    names.iter().all(|name| tensors.contains(name))
}

fn dense_swiglu_layout(decoder: &DecoderConfig, tensors: &TensorCatalog) -> bool {
    dense_text_root(tensors).is_some()
        && (decoder.tie_word_embeddings || tensors.contains("lm_head.weight"))
}

fn hybrid_linear_moe_layout(tensors: &TensorCatalog) -> bool {
    has_all(tensors, HYBRID_LINEAR_MOE_LAYOUT)
}

fn dense_attention_feature(tensors: &TensorCatalog) -> AttentionFeature {
    if ["model.layers.0", "layers.0"].into_iter().any(|root| {
        tensors.contains(&format!("{root}.self_attn.q_norm.weight"))
            && tensors.contains(&format!("{root}.self_attn.k_norm.weight"))
    }) {
        AttentionFeature::RmsNormalizedGroupedQuery
    } else {
        AttentionFeature::GroupedQuery
    }
}

fn dense_text_root(tensors: &TensorCatalog) -> Option<&'static str> {
    ["model.", ""].into_iter().find(|root| {
        [
            "embed_tokens.weight",
            "layers.0.self_attn.q_proj.weight",
            "layers.0.mlp.gate_proj.weight",
            "norm.weight",
        ]
        .iter()
        .all(|suffix| tensors.contains(&format!("{root}{suffix}")))
    })
}

fn invalid(message: impl Into<String>) -> ModelsError {
    ModelsError::InvalidConfig(message.into())
}

const HYBRID_MOE_LAYOUT: &[&str] = &[
    "language_model.model.embed_tokens.weight",
    "language_model.model.layers.0.router.proj.weight",
    "language_model.model.norm.weight",
];
const HYBRID_LINEAR_MOE_LAYOUT: &[&str] = &[
    "language_model.model.embed_tokens.weight",
    "language_model.model.layers.0.linear_attn.in_proj_qkv.weight",
    "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight",
    "language_model.model.norm.weight",
];
#[cfg(test)]
const DENSE_SWIGLU_LAYOUT: &[&str] = &[
    "model.embed_tokens.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.norm.weight",
];
#[cfg(test)]
const DENSE_QK_NORM_LAYOUT: &[&str] =
    &["model.layers.0.self_attn.q_norm.weight", "model.layers.0.self_attn.k_norm.weight"];

#[cfg(test)]
mod tests;
