use super::{
    super::{GatedFullAttention, rope},
    common_position,
};
use crate::engine::{Array, KvCache, Result, Stream, persistent_history};

impl GatedFullAttention {
    /// Attends over one persistent history per layer when the cohort allows it.
    pub(super) fn forward_persistent(
        &self,
        [queries, gate, keys, values]: [&Array; 4],
        caches: &mut [&mut KvCache],
        positions: &[i32],
        stream: &Stream,
    ) -> Result<Option<Array>> {
        let Some(attended) = persistent_history::attend(
            &persistent_history::Attend {
                queries,
                keys,
                values,
                scale: self.config.attention_scale,
                sinks: None,
            },
            caches,
            positions,
            |input| self.rotate_rows(input, positions, stream),
            stream,
        )?
        else {
            return Ok(None);
        };
        let shape =
            [i32::try_from(caches.len())?, 1, self.config.attention_heads * self.config.head_dim];
        let attended = attended.transpose(&[0, 2, 1, 3], stream)?.reshape(&shape, stream)?;
        let gate = gate.reshape(&shape, stream)?;
        self.project_output(&gate.sigmoid_mul(&attended, stream)?, stream).map(Some)
    }

    fn rotate_rows(&self, input: &Array, positions: &[i32], stream: &Stream) -> Result<Array> {
        if let Some(position) = common_position(positions) {
            return rope(input, &self.config, position, None, stream);
        }
        let offsets =
            positions.iter().map(|p| Ok(u32::try_from(*p)?)).collect::<Result<Vec<_>>>()?;
        let offsets = mirtal::Array::from_slice(&offsets, [offsets.len()])?;
        let input = input.transpose(&[0, 2, 1, 3], stream)?;
        Array::from_native(stream.native().graph().rope_batched(
            input.native(),
            mirtal::RopeOptions {
                dimensions: usize::try_from(self.config.rope_dimensions)?,
                traditional: false,
                base: Some(self.config.rope_base),
                scale: 1.0,
                offset: &offsets,
            },
        )?)
    }
}
