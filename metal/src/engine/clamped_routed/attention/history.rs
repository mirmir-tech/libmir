use super::ClampedRoutedAttention;
use crate::engine::{Array, DecoderCache, Error, KvCache, Result, Stream, persistent_history};

impl ClampedRoutedAttention {
    /// Attends over one persistent history per layer when the cohort allows it.
    /// Sliding layers keep their bounded ring and stay on per-row attention.
    pub(super) fn forward_persistent(
        &self,
        qkv: [&Array; 3],
        caches: &mut [&mut DecoderCache],
        positions: &[i32],
        layer: usize,
        stream: &Stream,
    ) -> Result<Option<Array>> {
        let mut layers = caches
            .iter_mut()
            .map(|cache| {
                cache
                    .attention_caches_mut()?
                    .get_mut(layer)
                    .ok_or_else(|| Error::InvalidModel("missing packed clamped cache layer".into()))
            })
            .collect::<Result<Vec<_>>>()?;
        self.attend_persistent(qkv, &mut layers, positions, stream)
    }

    /// `queries` and `keys` are `[B, 1, H, D]` and not yet rotated.
    pub(super) fn attend_persistent(
        &self,
        [queries, keys, values]: [&Array; 3],
        layers: &mut [&mut KvCache],
        positions: &[i32],
        stream: &Stream,
    ) -> Result<Option<Array>> {
        let offsets =
            positions.iter().map(|p| Ok(u32::try_from(*p)?)).collect::<Result<Vec<_>>>()?;
        let offsets = mirtal::Array::from_slice(&offsets, [offsets.len()])?;
        let values = values.transpose(&[0, 2, 1, 3], stream)?;
        let Some(attended) = persistent_history::attend(
            &persistent_history::Attend {
                queries,
                keys,
                values: &values,
                scale: self.config.scale,
                sinks: Some(&self.sinks),
            },
            layers,
            positions,
            |input| self.rotated_batch(input, &offsets, stream)?.transpose(&[0, 2, 1, 3], stream),
            stream,
        )?
        else {
            return Ok(None);
        };
        self.project_attended(&attended, i32::try_from(positions.len())?, 1, stream)
            .map(Some)
    }
}
