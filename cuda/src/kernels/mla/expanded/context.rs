//! The expanded keys, values and queries of one launch.

use mircuda::{CublasGemmOffsets, DeviceBuffer, LaunchConfig, Stream};

use super::{
    CONTEXT_ROWS, LatentExpandedRow, LatentExpansion, LatentExpansionBuffers, execute::linear,
};
use crate::{
    Result,
    kernels::{
        geometry::{narrow, product},
        mla::{
            LatentAttention, LatentAttentionInputs, LatentAttentionLayer, LatentAttentionWorkspace,
            LatentUpLayout,
        },
    },
};

impl LatentAttention {
    /// Lays the sequences out in pages, gathers their cache rows and
    /// expands them into every head's keys and values.
    pub(super) fn expand_context(
        &self,
        stream: &Stream,
        expansion: &mut LatentExpansion,
        layer: &LatentAttentionLayer<'_>,
        (pages, block_tables, table_stride): (&DeviceBuffer<u8>, &DeviceBuffer<u32>, usize),
        (rows, contexts): (&[LatentExpandedRow], &DeviceBuffer<u32>),
        buffers: &mut LatentExpansionBuffers,
    ) -> Result<()> {
        let spec = self.spec;
        let kernels = &self.expanded;
        let longest = rows.iter().map(|row| row.context()).max().unwrap_or(0);
        let stride = longest.div_ceil(CONTEXT_ROWS);
        let padded = rows.iter().map(|row| row.context().div_ceil(CONTEXT_ROWS)).sum::<usize>()
            * CONTEXT_ROWS;
        let sequences = narrow(rows.len())?;
        let config = LaunchConfig {
            grid: (1, 1, 1),
            block: (256, 1, 1),
            shared_memory_bytes: 0,
        };
        let layout =
            (contexts, &mut buffers.regions, &mut buffers.table, sequences, narrow(stride)?);
        kernels.layout.launch(stream, config, layout)?;
        let vectors = product(stride * CONTEXT_ROWS, spec.row_width() / 8)?;
        let config = LaunchConfig {
            grid: (narrow(vectors.div_ceil(256))?, sequences, 1),
            block: (256, 1, 1),
            shared_memory_bytes: 0,
        };
        let paging = (narrow(table_stride)?, narrow(spec.block_size)?);
        let gather = (pages, block_tables, contexts, &buffers.regions, &mut buffers.latent);
        kernels.gather.launch(
            stream,
            config,
            (gather.0, gather.1, gather.2, gather.3, gather.4, paging.0, paging.1),
        )?;
        let value_offset = match expansion.layout {
            LatentUpLayout::Fused => product(spec.nope_dim, spec.latent_rank)?,
            LatentUpLayout::Split => 0,
        };
        let head_dim = expansion.head_dim;
        let [key_plan, value_plan] = expansion.plans(stream, padded)?;
        let (key_weights, value_weights) = (layer.up.key(), layer.up.value());
        key_plan.execute(
            stream,
            (&buffers.latent, key_weights, &mut buffers.keys),
            CublasGemmOffsets::default(),
            (1.0, 0.0),
        )?;
        let offsets = CublasGemmOffsets {
            right: value_offset,
            ..CublasGemmOffsets::default()
        };
        value_plan.execute(
            stream,
            (&buffers.latent, value_weights, &mut buffers.values),
            offsets,
            (1.0, 0.0),
        )?;
        kernels.fill.launch(
            stream,
            linear(product(padded * spec.heads, head_dim - spec.nope_dim)?)?,
            (
                &buffers.latent,
                &mut buffers.keys,
                &mut buffers.values,
                narrow(padded)?,
                narrow(spec.heads)?,
                narrow(spec.nope_dim)?,
                narrow(spec.value_dim)?,
                narrow(head_dim)?,
            ),
        )?;
        Ok(())
    }

    /// Writes the expanded queries of `tokens` rows.
    pub(super) fn expanded_queries(
        &self,
        stream: &Stream,
        head_dim: usize,
        layer: &LatentAttentionLayer<'_>,
        inputs: &LatentAttentionInputs<'_>,
        tokens: usize,
        (workspace, buffers): (&LatentAttentionWorkspace, &mut LatentExpansionBuffers),
    ) -> Result<()> {
        let spec = self.spec;
        #[allow(clippy::cast_possible_truncation)]
        let (beta, window) = match layer.temperature {
            Some(temperature) => (temperature.beta as f32, narrow(temperature.window.get())?),
            None => (0.0, 0),
        };
        self.expanded.query.launch(
            stream,
            linear(product(tokens * spec.heads, head_dim)?)?,
            (
                inputs.query,
                &workspace.query,
                inputs.positions,
                &mut buffers.queries,
                narrow(tokens)?,
                narrow(spec.heads)?,
                narrow(spec.nope_dim)?,
                narrow(head_dim)?,
                beta,
                window,
            ),
        )?;
        Ok(())
    }
}
