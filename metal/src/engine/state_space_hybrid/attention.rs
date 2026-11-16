//! Plain softmax attention over the shared K/V cache: separate Q/K/V
//! projections with neither rotary embeddings nor Q/K normalization.

use models::{layout::DecoderConfig, weights::AttentionProjectionBindings};

use crate::engine::{
    Array, DecoderCache, Error, KvCache, ModelTensors, PagedContextMode, Result, Stream,
    attention::{AttentionBias, AttentionRequest},
    binding::BoundLinear,
    paged_attention_min_context,
};

#[derive(Debug)]
pub(super) struct PlainAttention {
    query: BoundLinear,
    key: BoundLinear,
    value: BoundLinear,
    output: BoundLinear,
    heads: i32,
    key_value_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl PlainAttention {
    pub(super) fn load(
        tensors: &ModelTensors,
        bindings: AttentionProjectionBindings<'_>,
        decoder: &DecoderConfig,
        stream: &Stream,
    ) -> Result<Self> {
        let scale = match decoder.attention_scale {
            Some(scale) => scale.to_string().parse()?,
            None => decoder.head_dim.to_string().parse::<f32>()?.sqrt().recip(),
        };
        Ok(Self {
            query: BoundLinear::load(tensors, bindings.query, stream)?,
            key: BoundLinear::load(tensors, bindings.key, stream)?,
            value: BoundLinear::load(tensors, bindings.value, stream)?,
            output: BoundLinear::load(tensors, bindings.output, stream)?,
            heads: i32::try_from(decoder.num_attention_heads)?,
            key_value_heads: i32::try_from(decoder.num_key_value_heads)?,
            head_dim: i32::try_from(decoder.head_dim)?,
            scale,
        })
    }

    fn project(
        &self,
        input: &Array,
        rows: i32,
        sequence: i32,
        stream: &Stream,
    ) -> Result<(Array, Array, Array)> {
        let project = |projection: &BoundLinear, heads| {
            projection
                .forward(input, stream)?
                .reshape(&[rows, sequence, heads, self.head_dim], stream)?
                .transpose(&[0, 2, 1, 3], stream)
        };
        Ok((
            project(&self.query, self.heads)?,
            project(&self.key, self.key_value_heads)?,
            project(&self.value, self.key_value_heads)?,
        ))
    }

    /// Attends `[1, sequence, hidden]` over one cache.
    pub(super) fn forward(
        &self,
        input: &Array,
        cache: &mut KvCache,
        causal: bool,
        stream: &Stream,
    ) -> Result<Array> {
        let shape = input.shape()?;
        let (rows, sequence) = (shape[0], shape[1]);
        let (queries, keys, values) = self.project(input, rows, sequence, stream)?;
        let attended = self.attend(&queries, &keys, &values, cache, causal, stream)?;
        self.finish(&attended, rows, sequence, stream)
    }

    /// Attends `[rows, sequence, hidden]`, one row per cache.
    pub(super) fn forward_packed(
        &self,
        input: &Array,
        caches: &mut [&mut DecoderCache],
        layer: usize,
        causal: bool,
        stream: &Stream,
    ) -> Result<Array> {
        let shape = input.shape()?;
        let (rows, sequence) = (shape[0], shape[1]);
        if usize::try_from(rows)? != caches.len() {
            return Err(Error::InvalidModel("packed attention rows and caches differ".into()));
        }
        let (queries, keys, values) = self.project(input, rows, sequence, stream)?;
        let sequence_usize = usize::try_from(sequence)?;
        let head_dim = usize::try_from(self.head_dim)?;
        let row = |array: &Array, index: usize, heads: i32| -> Result<Array> {
            array.slice(
                &[index, 0, 0, 0],
                &[index + 1, usize::try_from(heads)?, sequence_usize, head_dim],
                stream,
            )
        };
        let attended = caches
            .iter_mut()
            .enumerate()
            .map(|(index, cache)| {
                let queries = row(&queries, index, self.heads)?;
                let keys = row(&keys, index, self.key_value_heads)?;
                let values = row(&values, index, self.key_value_heads)?;
                let cache = cache.full_attention_cache(layer)?;
                self.attend(&queries, &keys, &values, cache, causal, stream)
            })
            .collect::<Result<Vec<_>>>()?;
        let attended = Array::concatenate(&attended.iter().collect::<Vec<_>>(), 0, stream)?;
        self.finish(&attended, rows, sequence, stream)
    }

    fn attend(
        &self,
        queries: &Array,
        keys: &Array,
        values: &Array,
        cache: &mut KvCache,
        causal: bool,
        stream: &Stream,
    ) -> Result<Array> {
        let context = cache.update_for_attention_mode(
            keys,
            values,
            stream,
            paged_attention_min_context(stream),
            PagedContextMode::View,
        )?;
        AttentionRequest::new(queries, &context, self.scale, causal, AttentionBias::None)?
            .execute(stream)
    }

    fn finish(&self, attended: &Array, rows: i32, sequence: i32, stream: &Stream) -> Result<Array> {
        let flat = attended
            .transpose(&[0, 2, 1, 3], stream)?
            .reshape(&[rows, sequence, self.heads * self.head_dim], stream)?;
        self.output.forward(&flat, stream)
    }
}
