use crate::engine::{Array, NormWeight, QuantizedLinear};

mod load;

#[derive(Debug)]
pub(super) struct AttentionWeights {
    pub(super) query: QuantizedLinear,
    pub(super) key: QuantizedLinear,
    pub(super) value: Option<QuantizedLinear>,
    pub(super) output: QuantizedLinear,
    pub(super) query_norm: NormWeight,
    pub(super) key_norm: NormWeight,
    pub(super) rope_frequencies: Option<Array>,
}

#[derive(Debug)]
pub(super) struct DenseWeights {
    pub(super) gate: QuantizedLinear,
    pub(super) up: QuantizedLinear,
    pub(super) down: QuantizedLinear,
}

#[derive(Debug)]
pub(super) struct RouterWeights {
    pub(super) projection: QuantizedLinear,
    pub(super) norm_scale: Array,
    pub(super) expert_scale: Array,
}

#[derive(Debug)]
pub(super) struct ExpertWeights {
    pub(super) gate: QuantizedLinear,
    pub(super) up: QuantizedLinear,
    pub(super) down: QuantizedLinear,
}

#[derive(Debug)]
pub(super) struct LayerWeights {
    pub(super) input_norm: NormWeight,
    pub(super) post_attention_norm: NormWeight,
    pub(super) pre_dense_norm: NormWeight,
    pub(super) post_dense_norm: NormWeight,
    pub(super) pre_expert_norm: NormWeight,
    pub(super) post_expert_norm: NormWeight,
    pub(super) post_feed_forward_norm: NormWeight,
    pub(super) layer_scalar: Array,
    pub(super) attention: AttentionWeights,
    pub(super) dense: DenseWeights,
    pub(super) router: RouterWeights,
    pub(super) experts: ExpertWeights,
}
