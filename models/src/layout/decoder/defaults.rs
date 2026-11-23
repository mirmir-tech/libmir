//! Checkpoint import defaults, not backend selection. Explicit configuration
//! and the complete mir-model-spec.toml take precedence over these values.
use serde_json::Value;

use crate::{error::Result, semantic::QkNormalization};

#[derive(Default)]
pub(super) struct Defaults {
    pub attention_scale: Option<f64>,
    pub qk_normalization: Option<QkNormalization>,
    pub layer_norm: bool,
    pub parallel_block: bool,
    pub attention_pattern: AttentionPatternDefault,
    /// The expert scoring function a schema fixes instead of declaring
    /// `scoring_func` (GLM-4.5's sigmoid router).
    pub expert_scoring: Option<&'static str>,
    pub router: Router,
    embedding: Embedding,
}

impl Defaults {
    /// The expert router; the sparsemixer's jitter defaults to transformers
    /// `PhimoeConfig`'s 0.01.
    pub fn expert_router(&self, decoder: &Value) -> Result<super::ExpertRouter> {
        Ok(match self.router {
            Router::SoftmaxTopK => super::ExpertRouter::SoftmaxTopK,
            Router::SigmoidInputScaled => super::ExpertRouter::SigmoidInputScaled,
            Router::SparseMixer => {
                let jitter =
                    super::parse::optional_f64(decoder, &["router_jitter_noise"])?.unwrap_or(0.01);
                if !jitter.is_finite() || jitter < 0.0 {
                    return Err(super::parse::invalid(
                        "router_jitter_noise must be finite and non-negative",
                    ));
                }
                super::ExpertRouter::SparseMixer { jitter }
            },
        })
    }

    pub const fn norm_kind(&self) -> crate::layout::NormKind {
        if self.layer_norm {
            crate::layout::NormKind::LayerNorm
        } else {
            crate::layout::NormKind::RmsNorm
        }
    }
}

/// Which per-layer attention traits a schema has.
#[derive(Default, Clone, Copy, PartialEq, Eq)]
pub(super) enum AttentionPatternDefault {
    #[default]
    Uniform,
    /// Only sliding layers rotate (Cohere 2).
    SlidingRotary,
    /// Llama 4's rotation-free layers, chunks, temperature and unit Q/K norms.
    Llama4,
}

/// How a schema's routed experts are selected when it does not say.
#[derive(Default, Clone, Copy, PartialEq, Eq)]
pub(super) enum Router {
    #[default]
    SoftmaxTopK,
    /// Sparsemixer with `router_jitter_noise` (Phi-3.5-MoE).
    SparseMixer,
    /// Top logits whose sigmoid scales the expert input (Llama 4).
    SigmoidInputScaled,
}

#[derive(Default)]
enum Embedding {
    #[default]
    Unspecified,
    SqrtHiddenSize,
}

impl Defaults {
    pub fn normalization(&self, config: &Value) -> Result<Option<QkNormalization>> {
        Ok(config
            .get("qk_normalization")
            .filter(|value| !value.is_null())
            .map(|value| serde_json::from_value(value.clone()))
            .transpose()?
            .or(self.qk_normalization))
    }

    pub fn embedding_multiplier(&self, hidden_size: usize) -> Result<Option<f64>> {
        Ok(match self.embedding {
            Embedding::Unspecified => None,
            Embedding::SqrtHiddenSize => Some(f64::from(u32::try_from(hidden_size)?).sqrt()),
        })
    }
}

pub(super) fn for_checkpoint(config: &Value) -> Defaults {
    // Gemma 4: established by our Metal 26B reference runs and the CUDA
    // 12B BF16 regression. See docs/semantic-model-migration.md for provenance.
    match config.get("model_type").and_then(Value::as_str) {
        Some("gemma4" | "gemma4_text" | "gemma4_unified") => Defaults {
            attention_scale: Some(1.0),
            qk_normalization: Some(QkNormalization::QueryKeyValueRms),
            embedding: Embedding::SqrtHiddenSize,
            ..Defaults::default()
        },
        // Cohere's `LayerNorm` without bias, one norm feeding attention and
        // the MLP in parallel (transformers `CohereDecoderLayer`); Cohere2
        // rotates its sliding layers only (`Cohere2Attention`).
        // transformers `PhimoeDecoderLayer`: `LayerNorm` with bias, and the
        // sparsemixer router.
        Some("phimoe") => Defaults {
            layer_norm: true,
            router: Router::SparseMixer,
            ..Defaults::default()
        },
        // transformers `Llama4TextMoe`: a sigmoid-scaled input to one expert,
        // beside an ungated shared expert.
        Some("llama4_text" | "llama4") => Defaults {
            attention_pattern: AttentionPatternDefault::Llama4,
            router: Router::SigmoidInputScaled,
            ..Defaults::default()
        },
        // transformers `Glm4MoeTopkRouter` scores by sigmoid.
        Some("glm4_moe") => Defaults {
            expert_scoring: Some("sigmoid"),
            ..Defaults::default()
        },
        Some("cohere") => Defaults {
            layer_norm: true,
            parallel_block: true,
            ..Defaults::default()
        },
        Some("cohere2") => Defaults {
            layer_norm: true,
            parallel_block: true,
            attention_pattern: AttentionPatternDefault::SlidingRotary,
            ..Defaults::default()
        },
        _ => Defaults::default(),
    }
}
