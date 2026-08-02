use models::weights::RoutedDecoderLayerBindings;

use super::{config::ClampedRoutedConfig, projection::BoundLinear};
use crate::engine::{
    Array, KvCache, ModelTensors, NormWeight, PagedContextMode, Result, Stream,
    paged_attention_min_context,
};

#[derive(Debug)]
pub(super) struct ClampedRoutedAttention {
    norm: NormWeight,
    query: BoundLinear,
    key: BoundLinear,
    value: BoundLinear,
    output: BoundLinear,
    sinks: Array,
    frequencies: Array,
    config: ClampedRoutedConfig,
}

impl ClampedRoutedAttention {
    pub fn load(
        tensors: &ModelTensors,
        bindings: RoutedDecoderLayerBindings<'_>,
        config: ClampedRoutedConfig,
        stream: &Stream,
    ) -> Result<Self> {
        Ok(Self {
            norm: NormWeight::load_name(tensors, &bindings.input_norm.source)?,
            query: BoundLinear::load(tensors, bindings.query, stream)?,
            key: BoundLinear::load(tensors, bindings.key, stream)?,
            value: BoundLinear::load(tensors, bindings.value, stream)?,
            output: BoundLinear::load(tensors, bindings.attention_output, stream)?,
            sinks: tensors.get(&bindings.attention_sinks.source)?,
            frequencies: Array::yarn_rope_frequencies(
                config.head_dim,
                config.rope_base,
                config.rope_factor,
                config.beta_fast,
                config.beta_slow,
                config.original_context,
                stream,
            )?,
            config,
        })
    }

    pub fn forward(
        &self,
        input: &Array,
        cache: &mut KvCache,
        position: i32,
        causal: bool,
        stream: &Stream,
    ) -> Result<Array> {
        let sequence = *input.shape()?.get(1).ok_or_else(|| {
            crate::engine::Error::InvalidModel(
                "clamped-routed attention input has no sequence axis".into(),
            )
        })?;
        let hidden = self.norm.apply(input, self.config.epsilon, stream)?;
        let queries = self
            .query
            .forward(&hidden, stream)?
            .reshape(&[1, sequence, self.config.heads, self.config.head_dim], stream)?;
        let keys = self
            .key
            .forward(&hidden, stream)?
            .reshape(&[1, sequence, self.config.kv_heads, self.config.head_dim], stream)?;
        let values = self
            .value
            .forward(&hidden, stream)?
            .reshape(&[1, sequence, self.config.kv_heads, self.config.head_dim], stream)?;
        let queries = self.rope(&queries, position, stream)?;
        let keys = self.rope(&keys, position, stream)?;
        let values = values.transpose(&[0, 2, 1, 3], stream)?;
        let context = cache.update_for_attention_mode(
            &keys,
            &values,
            stream,
            paged_attention_min_context(stream),
            PagedContextMode::View,
        )?;
        let attended = match context.mask.as_ref() {
            Some(mask) => queries.masked_scaled_dot_product_attention_with_sinks(
                &context.keys,
                &context.values,
                self.config.scale,
                mask,
                &self.sinks,
                stream,
            )?,
            None => queries.scaled_dot_product_attention_with_sinks(
                &context.keys,
                &context.values,
                self.config.scale,
                causal,
                &self.sinks,
                stream,
            )?,
        };
        let output = attended
            .transpose(&[0, 2, 1, 3], stream)?
            .reshape(&[1, sequence, self.config.heads * self.config.head_dim], stream)?;
        self.output.forward(&output, stream)
    }

    fn rope(&self, input: &Array, position: i32, stream: &Stream) -> Result<Array> {
        input
            .transpose(&[0, 2, 1, 3], stream)?
            .rope_with_frequencies(
                self.config.head_dim,
                false,
                &self.frequencies,
                position,
                stream,
            )?
            .multiply_scalar(self.config.rope_concentration, stream)
    }
}
