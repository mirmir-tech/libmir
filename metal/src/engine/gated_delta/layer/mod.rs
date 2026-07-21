mod decode;
mod load;
mod projection;

use models::layout::LinearAttentionConfig;

use self::{decode::CompiledDecode, projection::split_qkv};
use crate::engine::{
    Array, Error, GatedDeltaInputs, GatedDeltaState, NormWeight, QuantizedLinear, Result, Stream,
    fusion_planner::FusionPlanner,
};

#[derive(Debug, Clone, Copy)]
pub struct GatedDeltaLayerConfig {
    key_heads: i32,
    value_heads: i32,
    key_head_dim: i32,
    value_head_dim: i32,
    rms_norm_eps: f32,
}

#[derive(Debug)]
pub struct GatedDeltaLayer {
    config: GatedDeltaLayerConfig,
    in_proj_qkv: QuantizedLinear,
    in_proj_z: QuantizedLinear,
    in_proj_b: QuantizedLinear,
    in_proj_a: QuantizedLinear,
    out_proj: QuantizedLinear,
    conv_weight: Array,
    norm_weight: NormWeight,
    a_log: Array,
    dt_bias: Array,
    compiled_decode: Option<CompiledDecode>,
}

impl GatedDeltaLayerConfig {
    pub fn from_linear_attention(
        linear: &LinearAttentionConfig,
        rms_norm_eps: f64,
    ) -> Result<Self> {
        let config = Self {
            key_heads: i32::try_from(linear.key_heads)?,
            value_heads: i32::try_from(linear.value_heads)?,
            key_head_dim: i32::try_from(linear.key_head_dim)?,
            value_head_dim: i32::try_from(linear.value_head_dim)?,
            rms_norm_eps: rms_norm_eps.to_string().parse()?,
        };
        if config.key_heads <= 0
            || config.value_heads <= 0
            || config.key_head_dim <= 0
            || config.value_head_dim <= 0
            || config.value_heads % config.key_heads != 0
            || !config.rms_norm_eps.is_finite()
        {
            return Err(Error::InvalidModel(format!(
                "invalid Gated Delta layer configuration: {config:?}"
            )));
        }
        Ok(config)
    }
}

impl GatedDeltaLayer {
    pub fn forward(
        &self,
        input: &Array,
        state: &mut GatedDeltaState,
        stream: &Stream,
    ) -> Result<Array> {
        let shape = input.shape()?;
        let batch = dimension(&shape, 0)?;
        let sequence = dimension(&shape, 1)?;
        let key_width = self
            .config
            .key_heads
            .checked_mul(self.config.key_head_dim)
            .ok_or(Error::ShapeOverflow)?;
        let value_width = self
            .config
            .value_heads
            .checked_mul(self.config.value_head_dim)
            .ok_or(Error::ShapeOverflow)?;
        let fusion = FusionPlanner::new(stream).gated_delta();
        if sequence == 1
            && fusion.compiled_normalized_decode
            && let Some(compiled) = self.compiled_decode.as_ref()
            && let Some(output) = compiled.forward(input, state, stream)?
        {
            return Ok(output);
        }
        let mixed = state.convolve_silu(
            &self.in_proj_qkv.forward(input, stream)?,
            &self.conv_weight,
            stream,
        )?;
        let (query, key, value) =
            split_qkv(&mixed, usize::try_from(key_width)?, usize::try_from(value_width)?, stream)?;
        let query = query
            .reshape(&[batch, sequence, self.config.key_heads, self.config.key_head_dim], stream)?;
        let key = key
            .reshape(&[batch, sequence, self.config.key_heads, self.config.key_head_dim], stream)?;
        let value = value.reshape(
            &[batch, sequence, self.config.value_heads, self.config.value_head_dim],
            stream,
        )?;
        let gate = self.in_proj_z.forward(input, stream)?.reshape(
            &[batch, sequence, self.config.value_heads, self.config.value_head_dim],
            stream,
        )?;
        let alpha = self.in_proj_a.forward(input, stream)?;
        let beta = self.in_proj_b.forward(input, stream)?;
        let inputs = GatedDeltaInputs {
            query: &query,
            key: &key,
            value: &value,
            alpha: &alpha,
            beta: &beta,
            a_log: &self.a_log,
            dt_bias: &self.dt_bias,
        };
        let recurrent = if sequence == 1 && fusion.recurrent_decode {
            if fusion.recurrent_normalization {
                state.update_normalized(inputs, stream)?
            } else {
                normalized_update(state, inputs, self.config.key_head_dim, true, stream)?
            }
        } else {
            normalized_update(state, inputs, self.config.key_head_dim, false, stream)?
        };
        let normalized = self.norm_weight.apply(&recurrent, self.config.rms_norm_eps, stream)?;
        let output = Array::from_native(stream.precise_silu_mul(
            recurrent.native(),
            gate.native(),
            normalized.native(),
        )?)?;
        self.out_proj
            .forward(&output.reshape(&[batch, sequence, value_width], stream)?, stream)
    }

    pub(crate) fn forward_packed(
        &self,
        input: &Array,
        states: &mut [&mut GatedDeltaState],
        stream: &Stream,
    ) -> Result<Option<Array>> {
        if !FusionPlanner::new(stream).gated_delta().compiled_normalized_decode {
            return Ok(None);
        }
        self.compiled_decode
            .as_ref()
            .map_or(Ok(None), |compiled| compiled.forward_batch(input, states, stream))
    }
}

fn normalized_update(
    state: &mut GatedDeltaState,
    inputs: GatedDeltaInputs<'_>,
    key_dimension: i32,
    fused: bool,
    stream: &Stream,
) -> Result<Array> {
    let inverse = key_dimension.to_string().parse::<f32>()?.sqrt().recip();
    let query = inputs
        .query
        .rms_norm_unit(1.0e-6, stream)?
        .multiply_scalar(inverse * inverse, stream)?;
    let key = inputs.key.rms_norm_unit(1.0e-6, stream)?.multiply_scalar(inverse, stream)?;
    let inputs = GatedDeltaInputs { query: &query, key: &key, ..inputs };
    if fused {
        state.update_fused(inputs, stream)
    } else {
        state.update(inputs, stream)
    }
}

fn dimension(shape: &[i32], axis: usize) -> Result<i32> {
    shape
        .get(axis)
        .copied()
        .ok_or_else(|| Error::InvalidModel("Gated Delta input rank is invalid".into()))
}

#[cfg(test)]
mod tests;
