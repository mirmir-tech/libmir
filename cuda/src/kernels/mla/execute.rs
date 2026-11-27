use mircuda::{DeviceBuffer, LaunchConfig, Stream, bf16};
use models::semantic::QueryTemperature;

use super::{
    HEADS_PER_BLOCK, LatentAbsorption, LatentAttention, LatentAttentionWorkspace, LatentSplit,
    LatentUpWeights, MMA_GROUPS, MMA_HEADS, MMA_THREADS, mma_shared,
};
use crate::{
    Error, Result,
    kernels::geometry::{narrow, product, require},
};

/// Layer constants and weights of one latent attention layer.
#[derive(Clone, Copy, Debug)]
pub struct LatentAttentionLayer<'a> {
    /// `[latent_rank]` `RMSNorm` weight of the cached latent.
    pub latent_norm: &'a DeviceBuffer<bf16>,
    /// Per-head key and value up-projections in the absorption plan's
    /// layout.
    pub up: LatentUpWeights<'a>,
    /// `[rope_dim / 2]` rotary inverse frequencies, already `YaRN`-scaled.
    pub inverse_frequencies: &'a DeviceBuffer<f32>,
    pub epsilon: f32,
    /// Factor on the rotary cosine and sine (`YaRN` `mscale` ratio).
    pub rope_scale: f32,
    pub softmax_scale: f32,
    /// Position-dependent query scale on top of `softmax_scale`.
    pub temperature: Option<QueryTemperature>,
}

/// The paged latent cache one launch reads and extends.
#[derive(Debug)]
pub struct LatentCacheView<'a> {
    /// `[blocks · block_size, latent_rank + rope_dim]` BF16 rows.
    pub pages: &'a mut DeviceBuffer<u8>,
    /// `[sequences, table_stride]` physical block of every logical block.
    pub block_tables: &'a DeviceBuffer<u32>,
    pub table_stride: usize,
}

/// Per-token device inputs; `tokens` is the absorption plan's token count.
#[derive(Clone, Copy, Debug)]
pub struct LatentAttentionInputs<'a> {
    /// `[tokens, heads, nope + rope]` projected queries.
    pub query: &'a DeviceBuffer<bf16>,
    /// `[tokens, latent_rank + rope_dim]` key/value down-projection.
    pub latent: &'a DeviceBuffer<bf16>,
    /// Position of every token in its sequence; it attends to the tokens up
    /// to and including it.
    pub positions: &'a DeviceBuffer<u32>,
    /// Cache row each token's latent is stored into; `u32::MAX` skips the
    /// store (padding rows, shared pages that already hold the token).
    pub slot_mapping: &'a DeviceBuffer<u32>,
    /// Block-table row of every token's sequence.
    pub query_sequences: &'a DeviceBuffer<u32>,
}

impl LatentAttention {
    /// Stores the tokens' latents, attends every query token over its visible
    /// cache rows and writes `[tokens, heads · value_dim]` to `output`.
    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &self,
        stream: &Stream,
        absorption: &mut LatentAbsorption,
        layer: &LatentAttentionLayer<'_>,
        cache: LatentCacheView<'_>,
        inputs: LatentAttentionInputs<'_>,
        workspace: &mut LatentAttentionWorkspace,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let tokens = absorption.tokens();
        self.validate(tokens, absorption, layer, &cache, &inputs, output)?;
        let LatentCacheView { pages, block_tables, table_stride } = cache;
        let spec = self.spec;
        self.prepare_rows(stream, tokens, layer, pages, &inputs, workspace)?;
        absorption.absorb_query(stream, inputs.query, &layer.up, workspace)?;
        #[allow(clippy::cast_possible_truncation)]
        let (temperature_beta, temperature_window) = match layer.temperature {
            Some(temperature) => (temperature.beta as f32, narrow(temperature.window.get())?),
            None => (0.0, 0),
        };
        // The warp kernel tiles (head groups of eight, tokens, splits); the
        // tensor-core kernels a flat list of (token, sixteen heads) row groups.
        let row_groups = |groups: usize| -> Result<(u32, u32, u32)> {
            let blocks = product(tokens, spec.heads / MMA_HEADS)?.div_ceil(groups);
            Ok((narrow(blocks)?, 1, narrow(spec.splits)?))
        };
        let (grid, threads, groups) = match &self.split {
            LatentSplit::Warp(_) => (
                (
                    narrow(spec.heads.div_ceil(HEADS_PER_BLOCK))?,
                    narrow(tokens)?,
                    narrow(spec.splits)?,
                ),
                HEADS_PER_BLOCK * 32,
                0,
            ),
            LatentSplit::TensorCore(_) => (row_groups(1)?, MMA_THREADS, 1),
            LatentSplit::TensorCorePairs(_) => {
                (row_groups(MMA_GROUPS)?, MMA_GROUPS * MMA_THREADS, MMA_GROUPS)
            },
        };
        let config = LaunchConfig {
            grid,
            block: (narrow(threads)?, 1, 1),
            shared_memory_bytes: if groups == 0 {
                0
            } else {
                narrow(mma_shared(spec.row_width(), groups))?
            },
        };
        let arguments = (
            &workspace.query,
            &*pages,
            block_tables,
            inputs.query_sequences,
            inputs.positions,
            &mut workspace.values,
            &mut workspace.statistics,
            narrow(tokens)?,
            narrow(spec.heads)?,
            narrow(table_stride)?,
            narrow(spec.block_size)?,
            layer.softmax_scale,
            temperature_beta,
            temperature_window,
        );
        match &self.split {
            LatentSplit::Warp(kernel) => kernel.launch(stream, config, arguments)?,
            LatentSplit::TensorCore(kernel) => kernel.launch(stream, config, arguments)?,
            LatentSplit::TensorCorePairs(kernel) => kernel.launch(stream, config, arguments)?,
        }
        let rows = product(tokens, spec.heads)?;
        self.merge.launch(
            stream,
            LaunchConfig {
                grid: (narrow(rows)?, 1, 1),
                block: (narrow(spec.latent_rank.min(512))?, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                &workspace.values,
                &workspace.statistics,
                &mut workspace.latent_output,
                narrow(rows)?,
                narrow(spec.splits)?,
            ),
        )?;
        absorption.expand_output(stream, &layer.up, workspace, output)
    }

    /// Stores every token's normalized latent and rotated key at its slot
    /// and writes the rotated query part into the absorbed query rows.
    pub(super) fn prepare_rows(
        &self,
        stream: &Stream,
        tokens: usize,
        layer: &LatentAttentionLayer<'_>,
        pages: &mut DeviceBuffer<u8>,
        inputs: &LatentAttentionInputs<'_>,
        workspace: &mut LatentAttentionWorkspace,
    ) -> Result<()> {
        let spec = self.spec;
        Ok(self.prepare.launch(
            stream,
            LaunchConfig {
                grid: (narrow(tokens)?, 1, 1),
                block: (256, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                inputs.query,
                inputs.latent,
                layer.latent_norm,
                inputs.positions,
                layer.inverse_frequencies,
                inputs.slot_mapping,
                &mut workspace.query,
                pages,
                narrow(tokens)?,
                narrow(spec.heads)?,
                narrow(spec.nope_dim)?,
                layer.epsilon,
                layer.rope_scale,
            ),
        )?)
    }

    pub(super) fn validate(
        &self,
        tokens: usize,
        absorption: &LatentAbsorption,
        layer: &LatentAttentionLayer<'_>,
        cache: &LatentCacheView<'_>,
        inputs: &LatentAttentionInputs<'_>,
        output: &DeviceBuffer<bf16>,
    ) -> Result<()> {
        let spec = self.spec;
        // The semantic spec validates the window and beta; the kernels have
        // no position offset.
        let temperature = layer.temperature.is_none_or(|temperature| temperature.offset == 0);
        if tokens == 0
            || tokens > spec.max_queries
            || !layer.softmax_scale.is_finite()
            || !temperature
        {
            return Err(Error::InvalidPagedKv("invalid latent attention launch"));
        }
        let rows = product(tokens, spec.heads)?;
        require(
            "latent query",
            product(rows, spec.nope_dim + spec.rope_dim)?,
            inputs.query.len(),
        )?;
        require("latent input", product(tokens, spec.row_width())?, inputs.latent.len())?;
        require("latent positions", tokens, inputs.positions.len())?;
        require("latent slot mapping", tokens, inputs.slot_mapping.len())?;
        require("latent query sequences", tokens, inputs.query_sequences.len())?;
        require("latent norm", spec.latent_rank, layer.latent_norm.len())?;
        absorption.check(&layer.up)?;
        let (key, value) = absorption.weight_lengths()?;
        require("latent key up-projection", key, layer.up.key().len())?;
        require("latent value up-projection", value, layer.up.value().len())?;
        require("latent rotary frequencies", spec.rope_dim / 2, layer.inverse_frequencies.len())?;
        if cache.table_stride == 0 {
            return Err(Error::InvalidPagedKv("latent block tables have no columns"));
        }
        require("latent block tables", cache.table_stride, cache.block_tables.len())?;
        require("latent pages", self.spec.page_bytes(1)?, cache.pages.len())?;
        require("latent output", product(tokens, spec.heads * spec.value_dim)?, output.len())
    }
}
