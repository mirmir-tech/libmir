use crate::engine::attention::{AttentionBias, AttentionRequest};
mod batch;
#[cfg(test)]
mod offsets;

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
    #[cfg(test)]
    key_value_join: Option<crate::engine::binding::pair::KeyValueJoin>,
    output: BoundLinear,
    sinks: Array,
    frequencies: Array,
    config: ClampedRoutedConfig,
}

impl ClampedRoutedAttention {
    fn project_packed(
        &self,
        normalized: &Array,
        batch: i32,
        sequence: i32,
        causal: bool,
        stream: &Stream,
    ) -> Result<(Array, Array, Array)> {
        let project = |projection: &BoundLinear, heads| {
            projection
                .forward(normalized, stream)?
                .reshape(&[batch, sequence, heads, self.config.head_dim], stream)
        };
        let queries = project(&self.query, self.config.heads)?;
        let (keys, values) = self.project_key_value(normalized, sequence, causal, stream)?;
        let shape = [batch, sequence, self.config.kv_heads, self.config.head_dim];
        let keys = keys.reshape(&shape, stream)?;
        let values = values.reshape(&shape, stream)?;
        Ok((queries, keys, values))
    }

    fn project_key_value(
        &self,
        input: &Array,
        sequence: i32,
        causal: bool,
        stream: &Stream,
    ) -> Result<(Array, Array)> {
        #[cfg(not(test))]
        let _ = (sequence, causal);
        #[cfg(test)]
        if !causal
            && sequence == 1
            && stream.config().diagnostics.key_value_projection
                == crate::config::KeyValueProjection::JoinedDecode
            && let Some(joined) = &self.key_value_join
        {
            return joined.forward(input, stream);
        }
        Ok((self.key.forward(input, stream)?, self.value.forward(input, stream)?))
    }

    #[cfg(test)]
    fn capture_projection_inputs(&self, input: &Array, stream: &Stream) -> Result<()> {
        use crate::engine::probe::{self, ProjectionPair, Stage};
        probe::detail(Stage::AttentionNorm, input)?;
        ProjectionPair::capture(&self.key, &self.value, input, stream)
    }

    pub fn load(
        tensors: &ModelTensors,
        bindings: RoutedDecoderLayerBindings<'_>,
        config: ClampedRoutedConfig,
        stream: &Stream,
    ) -> Result<Self> {
        let key = BoundLinear::load(tensors, bindings.key, stream)?;
        let value = BoundLinear::load(tensors, bindings.value, stream)?;
        Ok(Self {
            #[cfg(test)]
            key_value_join: crate::engine::binding::pair::KeyValueJoin::new(&key, &value, stream)?,
            norm: NormWeight::load_name(tensors, &bindings.input_norm.source)?,
            query: BoundLinear::load(tensors, bindings.query, stream)?,
            key,
            value,
            output: BoundLinear::load(tensors, bindings.attention_output, stream)?,
            sinks: tensors.get(&bindings.attention_sinks.source)?,
            frequencies: Array::yarn_rope_frequencies(
                config.head_dim,
                config.rope_base,
                config.rope_factor,
                config.beta_fast,
                config.beta_slow,
                config.original_context,
                config.rope_truncate,
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
        #[cfg(test)]
        crate::engine::probe::detail(crate::engine::probe::Stage::AttentionNorm, &hidden)?;
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
        #[cfg(test)]
        for (stage, array) in [
            (crate::engine::probe::Stage::Query, &queries),
            (crate::engine::probe::Stage::Key, &keys),
            (crate::engine::probe::Stage::Value, &values),
        ] {
            crate::engine::probe::detail(stage, array)?;
        }
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
        let attended = AttentionRequest::new(
            &queries,
            &context,
            self.config.scale,
            causal,
            AttentionBias::Sinks(&self.sinks),
        )?
        .execute(stream)?;
        let output = attended
            .transpose(&[0, 2, 1, 3], stream)?
            .reshape(&[1, sequence, self.config.heads * self.config.head_dim], stream)?;
        #[cfg(test)]
        crate::engine::probe::detail(crate::engine::probe::Stage::Attended, &output)?;
        let output = self.output.forward(&output, stream)?;
        #[cfg(test)]
        crate::engine::probe::detail(crate::engine::probe::Stage::AttentionProjection, &output)?;
        Ok(output)
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
