//! Multi-head latent attention (`DeepSeek-V2/V3`, Kimi, Mistral Small 4):
//! each token caches one normalized key/value latent and one rotary key
//! shared by every head, and attention runs absorbed in latent space.
//!
//! Queries absorb the key up-projection (`q_nope · W_UK`), so scores against
//! the cached `[latent ‖ k_pe]` rows are `q_abs · latent + q_pe · k_pe`; the
//! softmax-weighted latents leave through the value up-projection `W_UV`.
//! The cache stores each `kv_lora_rank + rope_dim` row as the key and value
//! halves of one head ([`LatentRow`]).

mod rotary;
mod row;
#[cfg(test)]
mod tests;
mod weights;

use models::{
    layout::DecoderConfig, semantic::LatentAttentionSpec, weights::LatentAttentionBindings,
};

use self::{
    rotary::LatentRotary,
    row::LatentRow,
    weights::{LatentQuery, LatentUp},
};
use crate::engine::{
    Array, KvCache, ModelTensors, NormWeight, PagedContextMode, Result, Stream,
    binding::BoundLinear, paged_attention_min_context,
};

#[derive(Debug)]
pub struct LatentAttention {
    query: LatentQuery,
    key_value_down: BoundLinear,
    key_value_norm: NormWeight,
    up: LatentUp,
    output: BoundLinear,
    rotary: LatentRotary,
    row: LatentRow,
    heads: i32,
    nope: i32,
    rope: i32,
    value: i32,
    epsilon: f32,
    scale: f32,
}

impl LatentAttention {
    pub fn load(
        tensors: &ModelTensors,
        decoder: &DecoderConfig,
        index: usize,
        bindings: LatentAttentionBindings<'_>,
        stream: &Stream,
    ) -> Result<Self> {
        let spec = LatentAttentionSpec::for_layer(decoder, index)?;
        let epsilon: f32 = spec.latent_normalization.epsilon.to_string().parse()?;
        Ok(Self {
            query: LatentQuery::load(tensors, bindings.query, stream)?,
            key_value_down: BoundLinear::load(tensors, bindings.key_value_down, stream)?,
            key_value_norm: NormWeight::load_name(tensors, &bindings.key_value_norm.source)?,
            up: LatentUp::load(tensors, bindings.key_value_up, &spec, stream)?,
            output: BoundLinear::load(tensors, bindings.output, stream)?,
            rotary: LatentRotary::new(&spec)?,
            row: LatentRow::new(spec.key_value_rank, spec.rope_head_dim)?,
            heads: i32::try_from(spec.query_heads)?,
            nope: i32::try_from(spec.nope_head_dim)?,
            rope: i32::try_from(spec.rope_head_dim)?,
            value: i32::try_from(spec.value_head_dim)?,
            epsilon,
            scale: spec.scale.to_string().parse()?,
        })
    }

    /// Attends `[1, sequence, hidden]` normalized hidden states at
    /// `position` over one cache.
    pub fn forward(
        &self,
        input: &Array,
        cache: &mut KvCache,
        position: i32,
        causal: bool,
        stream: &Stream,
    ) -> Result<Array> {
        let shape = input.shape()?;
        let (rows, sequence) = (shape[0], shape[1]);
        // `[rows, heads, sequence, nope + rope]`.
        let query = self
            .query
            .forward(input, stream, self.epsilon)?
            .reshape(&[rows, sequence, self.heads, self.nope + self.rope], stream)?
            .transpose(&[0, 2, 1, 3], stream)?;
        let (nope, rope) = (usize::try_from(self.nope)?, usize::try_from(self.nope + self.rope)?);
        let stop = |end: usize| -> Result<Vec<usize>> {
            Ok(vec![
                usize::try_from(rows)?,
                usize::try_from(self.heads)?,
                usize::try_from(sequence)?,
                end,
            ])
        };
        let content_query = query.slice(&[0, 0, 0, 0], &stop(nope)?, stream)?;
        let rotary_query = query.slice(&[0, 0, 0, nope], &stop(rope)?, stream)?;
        let rotary_query = self.rotary.apply(&rotary_query, position, stream)?;
        // `[rows, 1, sequence, rank + rope]`: the normalized latent and the
        // rotated shared key of every new token.
        let compressed = self.key_value_down.forward(input, stream)?.expand_dims(&[1], stream)?;
        let (latent, rotary_key) = self.row.split(&compressed, stream)?;
        let latent = self.key_value_norm.apply(&latent, self.epsilon, stream)?;
        let rotary_key = self.rotary.apply(&rotary_key, position, stream)?;
        let new_rows =
            Array::concatenate(&[&latent, &rotary_key.astype_like(&latent, stream)?], 3, stream)?;
        let (keys, values) = self.row.halves(&new_rows, stream)?;
        let context = cache.update_for_attention_mode(
            &keys,
            &values,
            stream,
            paged_attention_min_context(stream),
            PagedContextMode::View,
        )?;
        let history = LatentRow::join(&context.keys, &context.values, stream)?;
        let (history_latent, _) = self.row.split(&history, stream)?;
        let absorbed = self.up.absorb_query(&content_query, stream)?;
        let queries = Array::concatenate(
            &[&absorbed, &rotary_query.astype_like(&absorbed, stream)?],
            3,
            stream,
        )?;
        let queries = self.rotary.temper(&queries, position, sequence, stream)?;
        let mask = match context.mask.as_ref() {
            Some(mask) => mirtal::AttentionMask::Array(mask.native()),
            None if causal && sequence > 1 => mirtal::AttentionMask::Causal,
            None => mirtal::AttentionMask::None,
        };
        // Scores against `[latent ‖ k_pe]`, values the latents themselves.
        let attended = Array::from_native(stream.native().graph().scaled_dot_product_attention(
            queries.native(),
            history.native(),
            history_latent.native(),
            mirtal::ScaledDotProductAttention { scale: self.scale, mask, sinks: None },
        )?)?;
        let output = self
            .up
            .expand_values(&attended, stream)?
            .transpose(&[0, 2, 1, 3], stream)?
            .reshape(&[rows, sequence, self.heads * self.value], stream)?;
        self.output.forward(&output, stream)
    }
}
