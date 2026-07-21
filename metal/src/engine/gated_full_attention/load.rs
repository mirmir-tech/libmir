use models::weights::GatedSoftmaxBindings;

use super::{GatedFullAttention, GatedFullAttentionConfig};
use crate::engine::{
    ModelTensors, NormWeight, QuantizedLinear, Result, Stream,
    binding::{adjusted_norm, affine_linear},
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
            query: affine_linear(tensors, bindings.query)?,
            key: affine_linear(tensors, bindings.key)?,
            value: affine_linear(tensors, bindings.value)?,
            output: affine_linear(tensors, bindings.output)?,
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
) -> Result<QuantizedLinear> {
    QuantizedLinear::load(tensors, &format!("{prefix}.{name}"), group_size)
}
