/// Tensors of one `ModernBERT` encoder layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EncoderTensor {
    /// Absent on layer 0, which attends to the embedding norm output directly.
    AttentionNorm,
    Qkv,
    AttentionOutput,
    MlpNorm,
    /// Fused `GeGLU` input and gate projection, input half first.
    MlpInput,
    MlpOutput,
}

/// Tensors of one pre-norm decision head layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum HeadTensor {
    AttentionNormWeight,
    AttentionNormBias,
    QkvWeight,
    QkvBias,
    AttentionOutputWeight,
    AttentionOutputBias,
    FeedForwardNormWeight,
    FeedForwardNormBias,
    UpWeight,
    UpBias,
    DownWeight,
    DownBias,
}

/// Tensors of the option scorer: norm, hidden projection, GELU, logit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ScorerTensor {
    NormWeight,
    NormBias,
    HiddenWeight,
    HiddenBias,
    LogitWeight,
    LogitBias,
}

/// Every tensor a Laya decision checkpoint contributes to inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DecisionTensor {
    TokenEmbedding,
    EmbeddingNorm,
    Encoder { layer: usize, tensor: EncoderTensor },
    FinalNorm,
    TypeEmbedding,
    Head { layer: usize, tensor: HeadTensor },
    Scorer(ScorerTensor),
}

/// Model widths that determine every tensor shape.
#[derive(Debug, Clone, Copy)]
pub(super) struct Widths {
    pub hidden: usize,
    pub intermediate: usize,
    pub vocab: usize,
}

impl DecisionTensor {
    /// Name of the tensor in `model.safetensors`.
    #[must_use]
    pub fn name(self) -> String {
        match self {
            Self::TokenEmbedding => "encoder.embeddings.tok_embeddings.weight".into(),
            Self::EmbeddingNorm => "encoder.embeddings.norm.weight".into(),
            Self::FinalNorm => "encoder.final_norm.weight".into(),
            Self::TypeEmbedding => "type_emb.weight".into(),
            Self::Encoder { layer, tensor } => {
                let suffix = match tensor {
                    EncoderTensor::AttentionNorm => "attn_norm.weight",
                    EncoderTensor::Qkv => "attn.Wqkv.weight",
                    EncoderTensor::AttentionOutput => "attn.Wo.weight",
                    EncoderTensor::MlpNorm => "mlp_norm.weight",
                    EncoderTensor::MlpInput => "mlp.Wi.weight",
                    EncoderTensor::MlpOutput => "mlp.Wo.weight",
                };
                format!("encoder.layers.{layer}.{suffix}")
            },
            Self::Head { layer, tensor } => format!("head.layers.{layer}.{}", head_suffix(tensor)),
            Self::Scorer(tensor) => match tensor {
                ScorerTensor::NormWeight => "scorer.0.weight",
                ScorerTensor::NormBias => "scorer.0.bias",
                ScorerTensor::HiddenWeight => "scorer.1.weight",
                ScorerTensor::HiddenBias => "scorer.1.bias",
                ScorerTensor::LogitWeight => "scorer.3.weight",
                ScorerTensor::LogitBias => "scorer.3.bias",
            }
            .into(),
        }
    }

    pub(super) fn shape(self, widths: Widths) -> Vec<usize> {
        let Widths { hidden, intermediate, vocab } = widths;
        match self {
            Self::TokenEmbedding => vec![vocab, hidden],
            Self::TypeEmbedding => vec![3, hidden],
            Self::EmbeddingNorm | Self::FinalNorm => vec![hidden],
            Self::Encoder { tensor, .. } => match tensor {
                EncoderTensor::AttentionNorm | EncoderTensor::MlpNorm => vec![hidden],
                EncoderTensor::Qkv => vec![3 * hidden, hidden],
                EncoderTensor::AttentionOutput => vec![hidden, hidden],
                EncoderTensor::MlpInput => vec![2 * intermediate, hidden],
                EncoderTensor::MlpOutput => vec![hidden, intermediate],
            },
            Self::Head { tensor, .. } => match tensor {
                HeadTensor::QkvWeight => vec![3 * hidden, hidden],
                HeadTensor::QkvBias => vec![3 * hidden],
                HeadTensor::AttentionOutputWeight => vec![hidden, hidden],
                HeadTensor::UpWeight => vec![4 * hidden, hidden],
                HeadTensor::UpBias => vec![4 * hidden],
                HeadTensor::DownWeight => vec![hidden, 4 * hidden],
                HeadTensor::AttentionNormWeight
                | HeadTensor::AttentionNormBias
                | HeadTensor::AttentionOutputBias
                | HeadTensor::FeedForwardNormWeight
                | HeadTensor::FeedForwardNormBias
                | HeadTensor::DownBias => vec![hidden],
            },
            Self::Scorer(tensor) => match tensor {
                ScorerTensor::HiddenWeight => vec![hidden, hidden],
                ScorerTensor::LogitWeight => vec![1, hidden],
                ScorerTensor::LogitBias => vec![1],
                ScorerTensor::NormWeight | ScorerTensor::NormBias | ScorerTensor::HiddenBias => {
                    vec![hidden]
                },
            },
        }
    }
}

const fn head_suffix(tensor: HeadTensor) -> &'static str {
    match tensor {
        HeadTensor::AttentionNormWeight => "norm1.weight",
        HeadTensor::AttentionNormBias => "norm1.bias",
        HeadTensor::QkvWeight => "self_attn.in_proj_weight",
        HeadTensor::QkvBias => "self_attn.in_proj_bias",
        HeadTensor::AttentionOutputWeight => "self_attn.out_proj.weight",
        HeadTensor::AttentionOutputBias => "self_attn.out_proj.bias",
        HeadTensor::FeedForwardNormWeight => "norm2.weight",
        HeadTensor::FeedForwardNormBias => "norm2.bias",
        HeadTensor::UpWeight => "linear1.weight",
        HeadTensor::UpBias => "linear1.bias",
        HeadTensor::DownWeight => "linear2.weight",
        HeadTensor::DownBias => "linear2.bias",
    }
}
