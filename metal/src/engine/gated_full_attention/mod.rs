mod load;

use models::layout::{AttentionOutput, DecoderConfig, RotaryEmbeddingLayout};

use super::{
    Array, Error, KvCache, NormWeight, PagedContextMode, Result, RopeOptions, Stream,
    attention::apply_mrope, binding::BoundLinear, fused_gate_up::split_last,
    native_paged_attention_mode,
};

#[derive(Debug, Clone)]
pub struct GatedFullAttentionConfig {
    attention_heads: i32,
    key_value_heads: i32,
    head_dim: i32,
    rope_dimensions: i32,
    rope_base: f32,
    attention_scale: f32,
    rms_norm_eps: f32,
    rope_layout: RotaryEmbeddingLayout,
}

#[derive(Debug)]
pub struct GatedFullAttention {
    config: GatedFullAttentionConfig,
    query: BoundLinear,
    key: BoundLinear,
    value: BoundLinear,
    output: BoundLinear,
    query_norm: NormWeight,
    key_norm: NormWeight,
}

impl GatedFullAttentionConfig {
    pub fn from_decoder(decoder: &DecoderConfig) -> Result<Self> {
        if decoder.attention_output != AttentionOutput::Gated {
            return Err(Error::InvalidModel("gated full attention requires an output gate".into()));
        }
        let head_dim = i32::try_from(decoder.head_dim)?;
        let partial = decoder.partial_rotary_factor.unwrap_or(1.0);
        let rope_dimensions: i32 = (f64::from(head_dim) * partial).round().to_string().parse()?;
        let config = Self {
            attention_heads: i32::try_from(decoder.num_attention_heads)?,
            key_value_heads: i32::try_from(decoder.num_key_value_heads)?,
            head_dim,
            rope_dimensions,
            rope_base: decoder.rope_theta.unwrap_or(10_000.0).to_string().parse()?,
            attention_scale: head_dim.to_string().parse::<f32>()?.sqrt().recip(),
            rms_norm_eps: decoder.rms_norm_eps.to_string().parse()?,
            rope_layout: decoder.rope_layout.clone(),
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<()> {
        if self.attention_heads <= 0
            || self.key_value_heads <= 0
            || self.head_dim <= 0
            || self.attention_heads % self.key_value_heads != 0
            || self.rope_dimensions <= 0
            || self.rope_dimensions > self.head_dim
            || self.rope_dimensions % 2 != 0
            || !self.rope_base.is_finite()
            || !self.attention_scale.is_finite()
            || !self.rms_norm_eps.is_finite()
        {
            return Err(Error::InvalidModel(format!(
                "invalid gated full attention configuration: {self:?}"
            )));
        }
        Ok(())
    }
}

impl GatedFullAttention {
    pub fn forward(
        &self,
        input: &Array,
        cache: &mut KvCache,
        page_min_context: usize,
        position: i32,
        causal: bool,
        stream: &Stream,
    ) -> Result<Array> {
        self.forward_with_positions(input, cache, page_min_context, position, causal, None, stream)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_positions(
        &self,
        input: &Array,
        cache: &mut KvCache,
        page_min_context: usize,
        position: i32,
        causal: bool,
        positions: Option<&Array>,
        stream: &Stream,
    ) -> Result<Array> {
        let shape = input.shape()?;
        let batch = dimension(&shape, 0, "batch")?;
        let sequence = dimension(&shape, 1, "sequence")?;
        let heads = self.config.attention_heads;
        let head_dim = self.config.head_dim;
        let query_width = heads.checked_mul(head_dim).ok_or(Error::ShapeOverflow)?;
        let projected = self.query.forward(input, stream)?.reshape(
            &[batch, sequence, heads, head_dim.checked_mul(2).ok_or(Error::ShapeOverflow)?],
            stream,
        )?;
        let (queries, gate) = split_last(&projected, usize::try_from(head_dim)?, stream)?;
        let gate = gate.reshape(&[batch, sequence, query_width], stream)?;
        let queries = self.query_norm.apply(&queries, self.config.rms_norm_eps, stream)?;
        let queries = rope(&queries, &self.config, position, positions, stream)?;

        let keys = self
            .key
            .forward(input, stream)?
            .reshape(&[batch, sequence, self.config.key_value_heads, head_dim], stream)?;
        let keys = self.key_norm.apply(&keys, self.config.rms_norm_eps, stream)?;
        let keys = rope(&keys, &self.config, position, positions, stream)?;
        let values = self
            .value
            .forward(input, stream)?
            .reshape(&[batch, sequence, self.config.key_value_heads, head_dim], stream)?
            .transpose(&[0, 2, 1, 3], stream)?;
        let mode = if sequence == 1 {
            native_paged_attention_mode(
                head_dim,
                heads,
                self.config.key_value_heads,
                usize::try_from(position)? + 1,
                stream.config().cache.force_native_paged_attention,
            )
        } else {
            PagedContextMode::View
        };
        let context =
            cache.update_for_attention_mode(&keys, &values, stream, page_min_context, mode)?;
        let attended = match context.paged {
            Some(paged) => queries.paged_scaled_dot_product_attention_with_scratch(
                paged.attention(),
                paged.scratch(),
                self.config.attention_scale,
                stream,
            )?,
            None => queries.scaled_dot_product_attention(
                &context.keys,
                &context.values,
                self.config.attention_scale,
                causal,
                stream,
            )?,
        };
        let attended = attended
            .transpose(&[0, 2, 1, 3], stream)?
            .reshape(&[batch, sequence, query_width], stream)?;
        self.output.forward(&gate.sigmoid_mul(&attended, stream)?, stream)
    }
}

fn rope(
    input: &Array,
    config: &GatedFullAttentionConfig,
    position: i32,
    positions: Option<&Array>,
    stream: &Stream,
) -> Result<Array> {
    let input = input.transpose(&[0, 2, 1, 3], stream)?;
    if let Some(positions) = positions {
        return apply_mrope(
            &input,
            positions,
            usize::try_from(config.rope_dimensions)?,
            config.rope_base,
            &config.rope_layout,
            stream,
        );
    }
    input.rope(
        RopeOptions {
            dimensions: config.rope_dimensions,
            traditional: false,
            base: Some(config.rope_base),
            scale: 1.0,
            offset: position,
        },
        stream,
    )
}

fn dimension(shape: &[i32], axis: usize, name: &str) -> Result<i32> {
    shape
        .get(axis)
        .copied()
        .ok_or_else(|| Error::InvalidModel(format!("attention input has no {name} axis")))
}

#[cfg(test)]
mod tests;
