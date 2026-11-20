use models::weights::{LinearAttentionBindings, TensorBinding};

use super::{CompiledDecode, GatedDeltaLayer, GatedDeltaLayerConfig};
use crate::engine::{
    Array, ModelTensors, NormWeight, QuantizedLinear, Result, Stream, binding::BoundLinear,
};

impl GatedDeltaLayer {
    pub fn load(
        tensors: &ModelTensors,
        prefix: &str,
        config: GatedDeltaLayerConfig,
        group_size: i32,
    ) -> Result<Self> {
        let norm_weight = NormWeight::load(tensors, &format!("{prefix}.norm"))?;
        Self::load_prefix(tensors, prefix, config, group_size, norm_weight, None)
    }

    /// The gated output norm multiplies by its weight in both checkpoint
    /// layouts: Hugging Face's zero-centred `(1 + w)` norms, which the caller
    /// shifts, are the layer and Q/K norms only (`mlx_lm` `qwen3_5.sanitize`).
    pub fn load_bindings(
        tensors: &ModelTensors,
        bindings: LinearAttentionBindings<'_>,
        config: GatedDeltaLayerConfig,
        stream: &Stream,
    ) -> Result<Self> {
        let mut layer = Self {
            config,
            in_proj_qkv: BoundLinear::load(tensors, bindings.qkv, stream)?,
            in_proj_z: BoundLinear::load(tensors, bindings.gate, stream)?,
            in_proj_b: BoundLinear::load(tensors, bindings.beta, stream)?,
            in_proj_a: BoundLinear::load(tensors, bindings.alpha, stream)?,
            out_proj: BoundLinear::load(tensors, bindings.output, stream)?,
            conv_weight: convolution(tensors, bindings.convolution, stream)?,
            norm_weight: NormWeight::load_name(tensors, &bindings.norm.source)?,
            a_log: tensors.get(&bindings.decay_log.source)?,
            dt_bias: tensors.get(&bindings.time_bias.source)?,
            compiled_decode: None,
        };
        layer.compiled_decode = CompiledDecode::new(&layer, stream)?;
        Ok(layer)
    }

    fn load_prefix(
        tensors: &ModelTensors,
        prefix: &str,
        config: GatedDeltaLayerConfig,
        group_size: i32,
        norm_weight: NormWeight,
        stream: Option<&Stream>,
    ) -> Result<Self> {
        let mut layer = Self {
            config,
            in_proj_qkv: linear(tensors, prefix, "in_proj_qkv", group_size)?,
            in_proj_z: linear(tensors, prefix, "in_proj_z", group_size)?,
            in_proj_b: linear(tensors, prefix, "in_proj_b", group_size)?,
            in_proj_a: linear(tensors, prefix, "in_proj_a", group_size)?,
            out_proj: linear(tensors, prefix, "out_proj", group_size)?,
            conv_weight: tensors.get(&format!("{prefix}.conv1d.weight"))?,
            norm_weight,
            a_log: tensors.get(&format!("{prefix}.A_log"))?,
            dt_bias: tensors.get(&format!("{prefix}.dt_bias"))?,
            compiled_decode: None,
        };
        if let Some(stream) = stream {
            layer.compiled_decode = CompiledDecode::new(&layer, stream)?;
        }
        Ok(layer)
    }
}

fn convolution(tensors: &ModelTensors, binding: &TensorBinding, stream: &Stream) -> Result<Array> {
    let weight = tensors.get(&binding.source)?;
    match binding.shape.as_slice() {
        [channels, 1, kernel] => {
            weight.reshape(&[i32::try_from(*channels)?, i32::try_from(*kernel)?, 1], stream)
        },
        _ => Ok(weight),
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
