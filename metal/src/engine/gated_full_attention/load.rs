use models::weights::GatedSoftmaxBindings;

use super::{GatedFullAttention, GatedFullAttentionConfig};
use crate::engine::{
    ModelTensors, NormWeight, QuantizedLinear, Result, Stream,
    binding::{BoundLinear, adjusted_norm},
};

impl GatedFullAttention {
    pub fn load_with_norm_shift(
        tensors: &ModelTensors,
        prefix: &str,
        config: GatedFullAttentionConfig,
        group_size: i32,
        norm_shift: f32,
        stream: &Stream,
    ) -> Result<Self> {
        Ok(Self {
            config,
            query: linear(tensors, prefix, "q_proj", group_size)?,
            key: linear(tensors, prefix, "k_proj", group_size)?,
            value: linear(tensors, prefix, "v_proj", group_size)?,
            output: linear(tensors, prefix, "o_proj", group_size)?,
            query_norm: NormWeight::load_adjusted(
                tensors,
                &format!("{prefix}.q_norm"),
                norm_shift,
                stream,
            )?,
            key_norm: NormWeight::load_adjusted(
                tensors,
                &format!("{prefix}.k_norm"),
                norm_shift,
                stream,
            )?,
        })
    }

    pub fn load_bindings(
        tensors: &ModelTensors,
        bindings: GatedSoftmaxBindings<'_>,
        config: GatedFullAttentionConfig,
        norm_shift: f32,
        stream: &Stream,
    ) -> Result<Self> {
        Ok(Self {
            config,
            query: BoundLinear::load(tensors, bindings.query, stream)?,
            key: BoundLinear::load(tensors, bindings.key, stream)?,
            value: BoundLinear::load(tensors, bindings.value, stream)?,
            output: BoundLinear::load(tensors, bindings.output, stream)?,
            query_norm: adjusted_norm(tensors, bindings.query_norm, norm_shift, stream)?,
            key_norm: adjusted_norm(tensors, bindings.key_norm, norm_shift, stream)?,
        })
    }
}

fn linear(
    tensors: &ModelTensors,
    prefix: &str,
    name: &str,
    group_size: i32,
) -> Result<BoundLinear> {
    QuantizedLinear::load(tensors, &format!("{prefix}.{name}"), group_size).map(BoundLinear::Affine)
}
