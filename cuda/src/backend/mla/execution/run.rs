//! One pass of the layer: norms, projections, latent attention (expanded
//! for one prefill sequence, absorbed otherwise) and the feed-forward tail.

use mircuda::{DeviceBuffer, bf16};

use super::{
    super::{LatentScratch, weights::QueryWeights},
    CudaLatentAttentionExecution, LatentQuery, Placement, Query, bf16_buffer,
};
use crate::{
    Error, PagedKvCache, Result,
    kernels::{LatentAttentionInputs, LatentAttentionLayer, LatentCacheView},
};

impl CudaLatentAttentionExecution {
    // The scratch guard is used by the last statement; the lint cannot see it.
    #[allow(clippy::significant_drop_tightening)]
    pub(super) fn run(
        &mut self,
        input: &DeviceBuffer<bf16>,
        pages: &mut PagedKvCache,
        placement: &Placement<'_>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let stream = self.backend.inner.stream.clone();
        let mut guard = self.scratch.lock()?;
        let scratch = &mut *guard;
        let latent_handle = self.latent_scratch.clone_handle();
        let mut latent_guard = latent_handle.lock()?;
        let LatentScratch {
            workspace,
            query: query_rows,
            latent,
            attended,
            query_latent,
            expansion: expansion_buffers,
        } = &mut *latent_guard;
        let weights = &self.layer.weights;
        self.input_norm.execute(
            &stream,
            input,
            bf16_buffer(&weights.input_norm)?,
            &mut scratch.normalized,
        )?;
        match &mut self.query {
            Query::Direct(projection) => {
                projection.execute(&scratch.normalized, query_rows)?;
            },
            Query::Latent(query) => {
                let LatentQuery { down, norm, up } = &mut **query;
                let (QueryWeights::Latent(query_weights), Some([raw, normalized])) =
                    (&weights.query, query_latent)
                else {
                    return Err(Error::InvalidDecoderKernel("query latent weights are missing"));
                };
                down.execute(&scratch.normalized, raw)?;
                norm.execute(&stream, raw, bf16_buffer(&query_weights.norm)?, normalized)?;
                up.execute(normalized, query_rows)?;
            },
        }
        self.key_value_down.execute(&scratch.normalized, latent)?;
        let layer = LatentAttentionLayer {
            latent_norm: bf16_buffer(&weights.key_value_norm)?,
            up: weights.up.weights(),
            inverse_frequencies: &self.layer.inverse_frequencies,
            epsilon: self.geometry.latent_epsilon,
            rope_scale: self.geometry.rope_scale,
            softmax_scale: self.geometry.softmax_scale,
            temperature: self.geometry.temperature,
        };
        let cache = LatentCacheView {
            pages: pages.latent_rows_mut()?,
            block_tables: placement.tables,
            table_stride: placement.table_stride,
        };
        let inputs = LatentAttentionInputs {
            query: query_rows,
            latent,
            positions: placement.positions,
            slot_mapping: placement.slot_mapping,
            query_sequences: placement.sequences,
        };
        // Sequences whose contexts fit the expansion and whose queries
        // repay expanding them attend expanded; the rest stays absorbed.
        match (placement.expanded, &mut self.expansion, expansion_buffers) {
            (Some(rows), Some(expansion), Some(buffers))
                if expansion.admits(buffers, rows.rows()) && repays(rows.rows()) =>
            {
                let scratch = (workspace, buffers);
                self.attention.execute_expanded(
                    &stream, expansion, &layer, cache, inputs, rows, scratch, attended,
                )?;
            },
            _ => self.attention.execute(
                &stream,
                &mut self.absorption,
                &layer,
                cache,
                inputs,
                workspace,
                attended,
            )?,
        }
        self.output.execute(attended, &mut scratch.attention)?;
        self.tail.execute(input, scratch, output)
    }
}

/// Whether expanding the sequences' contexts costs less than attending
/// absorbed: expansion pays once per context row, absorbed attention per
/// query and context row.
fn repays(rows: &[crate::kernels::LatentExpandedRow]) -> bool {
    let queries = rows.iter().map(|row| row.tokens).sum::<usize>();
    let contexts = rows.iter().map(|row| row.context()).sum::<usize>();
    queries * EXPANDED_QUERY_RATIO >= contexts
}

/// Context rows per query below which expanding repays. Over a 4096-token
/// prefix expanded prefill overtakes absorbed between 128- and 256-token
/// chunks on Mistral Small 4 (about 22 rows per query) and near 128 on
/// DeepSeek-V2-Lite (about 34).
const EXPANDED_QUERY_RATIO: usize = 24;
