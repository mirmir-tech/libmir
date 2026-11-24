//! One expanded launch: gather the sequences' cache rows, expand them per
//! head, build the queries and run the fused attention.

use mircuda::{DeviceBuffer, LaunchConfig, Stream, bf16};

use super::{CONTEXT_ROWS, LatentExpansion, LatentExpansionBuffers};
use crate::{
    Error, Result,
    kernels::{
        geometry::{narrow, product, require},
        mla::{
            LatentAttention, LatentAttentionInputs, LatentAttentionLayer, LatentAttentionWorkspace,
            LatentCacheView,
        },
    },
};

/// One sequence of an expanded launch: its `tokens` queries follow `start`
/// cached tokens.
#[derive(Clone, Copy, Debug)]
pub struct LatentExpandedRow {
    pub start: usize,
    pub tokens: usize,
}

impl LatentExpandedRow {
    /// Tokens the sequence's queries see.
    #[must_use]
    pub const fn context(self) -> usize {
        self.start + self.tokens
    }
}

/// Device metadata of packed sequences, as their paged prefill batch holds
/// it: packed query starts, context lengths and packed context starts.
#[derive(Clone, Copy, Debug)]
pub struct LatentExpandedMetadata<'a> {
    pub query_starts: &'a DeviceBuffer<u32>,
    pub contexts: &'a DeviceBuffer<u32>,
    pub context_starts: &'a DeviceBuffer<u32>,
}

/// The sequences of an expanded launch, in query order; sequence `s` uses
/// row `s` of the cache view's block tables.
#[derive(Clone, Copy, Debug)]
pub enum LatentExpandedRows<'a> {
    /// One sequence; the launch writes its attention metadata.
    One(LatentExpandedRow),
    /// Packed sequences described by their batch's metadata.
    Packed {
        rows: &'a [LatentExpandedRow],
        metadata: LatentExpandedMetadata<'a>,
    },
}

impl LatentExpandedRows<'_> {
    /// The sequences on the host.
    #[must_use]
    pub const fn rows(&self) -> &[LatentExpandedRow] {
        match self {
            Self::One(row) => std::slice::from_ref(row),
            Self::Packed { rows, .. } => rows,
        }
    }
}

impl LatentAttention {
    /// [`Self::execute`] for sequences whose contexts `expansion` holds
    /// ([`LatentExpansion::admits`]): stores the tokens' latents, then
    /// attends with expanded keys and values.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_expanded(
        &self,
        stream: &Stream,
        expansion: &mut LatentExpansion,
        layer: &LatentAttentionLayer<'_>,
        cache: LatentCacheView<'_>,
        inputs: LatentAttentionInputs<'_>,
        sequences: LatentExpandedRows<'_>,
        (workspace, buffers): (&mut LatentAttentionWorkspace, &mut LatentExpansionBuffers),
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let spec = self.spec;
        let rows = sequences.rows();
        let tokens = rows.iter().map(|row| row.tokens).sum::<usize>();
        if rows.iter().any(|row| row.tokens == 0)
            || tokens > spec.max_queries
            || !expansion.admits(buffers, rows)
        {
            return Err(Error::InvalidPagedKv("invalid expanded latent attention launch"));
        }
        let width = spec.nope_dim + spec.rope_dim;
        require("latent query", product(tokens * spec.heads, width)?, inputs.query.len())?;
        let value_width = product(tokens, spec.heads * spec.value_dim)?;
        require("expanded latent output", value_width, output.len())?;
        let LatentCacheView { pages, block_tables, table_stride } = cache;
        self.prepare_rows(stream, tokens, layer, pages, &inputs, workspace)?;
        if let LatentExpandedRows::One(row) = sequences {
            let config = LaunchConfig {
                grid: (1, 1, 1),
                block: (32, 1, 1),
                shared_memory_bytes: 0,
            };
            let (tokens, context) = (narrow(row.tokens)?, narrow(row.context())?);
            self.expanded
                .starts
                .launch(stream, config, (&mut buffers.starts, tokens, context))?;
        }
        let own = match sequences {
            LatentExpandedRows::One(_) => {
                let one = |range| buffers.starts.slice(range);
                Some([one(0..2)?, one(2..3)?, one(3..5)?])
            },
            LatentExpandedRows::Packed { .. } => None,
        };
        let metadata = match (&own, sequences) {
            (_, LatentExpandedRows::Packed { metadata, .. }) => metadata,
            (Some([query_starts, contexts, context_starts]), LatentExpandedRows::One(_)) => {
                LatentExpandedMetadata { query_starts, contexts, context_starts }
            },
            (None, LatentExpandedRows::One(_)) => {
                return Err(Error::InvalidPagedKv("expanded latent metadata is missing"));
            },
        };
        let longest = rows.iter().map(|row| row.context()).max().unwrap_or(0);
        let stride = longest.div_ceil(CONTEXT_ROWS);
        let paging = (&*pages, block_tables, table_stride);
        self.expand_context(stream, expansion, layer, paging, (rows, metadata.contexts), buffers)?;
        let head_dim = expansion.head_dim;
        self.expanded_queries(stream, head_dim, layer, &inputs, tokens, (workspace, buffers))?;
        let (keys, values) =
            (buffers.keys.reinterpret::<u8>()?, buffers.values.reinterpret::<u8>()?);
        // The fused attention takes exactly `tokens` query and output rows.
        let elements = product(tokens * spec.heads, head_dim)?;
        let queries = buffers.queries.slice(0..elements)?;
        let mut attended = buffers.output.as_ref().unwrap_or(output).slice(0..elements)?;
        expansion.attention.execute_paged_varlen(
            stream,
            &queries,
            &keys,
            &values,
            &mut attended,
            metadata.query_starts,
            metadata.contexts,
            metadata.context_starts,
            &buffers.table,
            &mut buffers.statistics,
            rows.len(),
            tokens,
            rows.iter().map(|row| row.tokens).max().unwrap_or(0),
            longest,
            stride,
            CONTEXT_ROWS,
            layer.softmax_scale,
        )?;
        if let Some(wide) = &buffers.output {
            self.expanded.output.launch(
                stream,
                linear(value_width)?,
                (
                    wide,
                    output,
                    narrow(tokens)?,
                    narrow(spec.heads)?,
                    narrow(spec.value_dim)?,
                    narrow(head_dim)?,
                ),
            )?;
        }
        Ok(())
    }
}

pub(super) fn linear(elements: usize) -> Result<LaunchConfig> {
    Ok(LaunchConfig {
        grid: (narrow(elements.div_ceil(256))?, 1, 1),
        block: (256, 1, 1),
        shared_memory_bytes: 0,
    })
}
