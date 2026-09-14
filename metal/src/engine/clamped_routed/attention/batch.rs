use super::ClampedRoutedAttention;
use crate::engine::{
    Array, DecoderCache, Error, PagedContextMode, Result, Stream,
    attention::{AttentionBias, AttentionRequest},
    paged_attention_min_context,
};

impl ClampedRoutedAttention {
    pub(in crate::engine::clamped_routed) fn forward_packed(
        &self,
        input: &Array,
        caches: &mut [&mut DecoderCache],
        positions: &[i32],
        layer: usize,
        causal: bool,
        stream: &Stream,
    ) -> Result<Array> {
        let (batch, sequence) = packed_shape(input, caches.len(), positions.len())?;
        let normalized = self.norm.apply(input, self.config.epsilon, stream)?;
        #[cfg(test)]
        self.capture_projection_inputs(&normalized, stream)?;
        let (queries, keys, values) =
            self.project_packed(&normalized, batch, sequence, causal, stream)?;
        #[cfg(test)]
        for (stage, array) in [
            (crate::engine::probe::Stage::Query, &queries),
            (crate::engine::probe::Stage::Key, &keys),
            (crate::engine::probe::Stage::Value, &values),
        ] {
            crate::engine::probe::detail(stage, array)?;
        }
        let sequence_usize = usize::try_from(sequence)?;
        let width = usize::try_from(self.config.head_dim)?;
        let row = |array: &Array, index, heads| {
            array.slice(&[index, 0, 0, 0], &[index + 1, sequence_usize, heads, width], stream)
        };
        #[cfg(test)]
        let rotated = if !causal
            && stream.config().diagnostics.rope_batching == crate::config::RopeBatching::Offsets
        {
            let offsets = positions
                .iter()
                .map(|&p| u32::try_from(p))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let offsets = mirtal::Array::from_slice(&offsets, [offsets.len()])?;
            Some((
                self.rotated_batch(&queries, &offsets, stream)?,
                self.rotated_batch(&keys, &offsets, stream)?,
            ))
        } else {
            None
        };
        let rotate_row = |array: &Array, index, heads, position, is_query| {
            #[cfg(not(test))]
            let _ = is_query;
            #[cfg(test)]
            if let Some((queries, keys)) = &rotated {
                return row(
                    if is_query {
                        queries
                    } else {
                        keys
                    },
                    index,
                    heads,
                )?
                .transpose(&[0, 2, 1, 3], stream);
            }
            self.rope(&row(array, index, heads)?, position, stream)
        };
        let mut attended = Vec::with_capacity(caches.len());
        for (index, (cache, position)) in caches.iter_mut().zip(positions).enumerate() {
            let query =
                rotate_row(&queries, index, usize::try_from(self.config.heads)?, *position, true)?;
            let key =
                rotate_row(&keys, index, usize::try_from(self.config.kv_heads)?, *position, false)?;
            let value = row(&values, index, usize::try_from(self.config.kv_heads)?)?
                .transpose(&[0, 2, 1, 3], stream)?;
            let context = cache
                .attention_caches_mut()?
                .get_mut(layer)
                .ok_or_else(|| Error::InvalidModel("missing packed clamped cache layer".into()))?
                .update_for_attention_mode(
                    &key,
                    &value,
                    stream,
                    paged_attention_min_context(stream),
                    PagedContextMode::View,
                )?;
            attended.push(
                AttentionRequest::new(
                    &query,
                    &context,
                    self.config.scale,
                    causal,
                    AttentionBias::Sinks(&self.sinks),
                )?
                .execute(stream)?,
            );
        }
        let rows = attended.iter().collect::<Vec<_>>();
        let output = Array::concatenate(&rows, 0, stream)?
            .transpose(&[0, 2, 1, 3], stream)?
            .reshape(&[batch, sequence, self.config.heads * self.config.head_dim], stream)?;
        #[cfg(test)]
        crate::engine::probe::detail(crate::engine::probe::Stage::Attended, &output)?;
        let output = self.output.forward(&output, stream)?;
        #[cfg(test)]
        crate::engine::probe::detail(crate::engine::probe::Stage::AttentionProjection, &output)?;
        Ok(output)
    }
}

fn packed_shape(input: &Array, rows: usize, positions: usize) -> Result<(i32, i32)> {
    let shape = input.shape()?;
    let [batch, sequence, _] = shape.as_slice() else {
        return Err(Error::InvalidModel("packed clamped input must have rank three".into()));
    };
    if usize::try_from(*batch)? != rows || positions != rows {
        return Err(Error::InvalidModel("packed clamped rows and positions differ".into()));
    }
    Ok((*batch, *sequence))
}
