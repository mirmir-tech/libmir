use mircuda::{DeviceBuffer, LaunchConfig, Stream, bf16};

use super::{HEADS_PER_BLOCK, LatentAbsorption, LatentAttention, LatentAttentionWorkspace};
use crate::{
    Error, Result,
    kernels::geometry::{narrow, product, require},
};

/// Layer constants and weights of one latent attention layer.
#[derive(Clone, Copy, Debug)]
pub struct LatentAttentionLayer<'a> {
    /// `[latent_rank]` `RMSNorm` weight of the cached latent.
    pub latent_norm: &'a DeviceBuffer<bf16>,
    /// `[heads · (nope + value), latent_rank]` key/value up-projection.
    pub key_value_up: &'a DeviceBuffer<bf16>,
    /// `[rope_dim / 2]` rotary inverse frequencies, already `YaRN`-scaled.
    pub inverse_frequencies: &'a DeviceBuffer<f32>,
    pub epsilon: f32,
    /// Factor on the rotary cosine and sine (`YaRN` `mscale` ratio).
    pub rope_scale: f32,
    pub softmax_scale: f32,
}

/// The paged latent cache one launch reads and extends.
#[derive(Debug)]
pub struct LatentCacheView<'a> {
    /// `[blocks · block_size, latent_rank + rope_dim]` BF16 rows.
    pub pages: &'a mut DeviceBuffer<u8>,
    /// `[sequences, max_blocks]` physical block of every logical block.
    pub block_tables: &'a DeviceBuffer<u32>,
}

/// Per-token device inputs; `tokens` is the absorption plan's token count.
#[derive(Clone, Copy, Debug)]
pub struct LatentAttentionInputs<'a> {
    /// `[tokens, heads, nope + rope]` projected queries.
    pub query: &'a DeviceBuffer<bf16>,
    /// `[tokens, latent_rank + rope_dim]` key/value down-projection.
    pub latent: &'a DeviceBuffer<bf16>,
    pub positions: &'a DeviceBuffer<u32>,
    /// Cache row each token's latent is stored into (`u32::MAX` skips it).
    pub slot_mapping: &'a DeviceBuffer<u32>,
    /// Block-table row of every token's sequence.
    pub query_sequences: &'a DeviceBuffer<u32>,
    /// Tokens of its sequence every query token attends to, itself included.
    pub query_visible: &'a DeviceBuffer<u32>,
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
        self.validate(tokens, layer, &cache, &inputs, output)?;
        let LatentCacheView { pages, block_tables } = cache;
        let spec = self.spec;
        self.prepare.launch(
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
                &mut *pages,
                narrow(tokens)?,
                narrow(spec.heads)?,
                narrow(spec.nope_dim)?,
                layer.epsilon,
                layer.rope_scale,
            ),
        )?;
        absorption.absorb_query(stream, inputs.query, layer.key_value_up, workspace)?;
        self.split.launch(
            stream,
            LaunchConfig {
                grid: (
                    narrow(spec.heads.div_ceil(HEADS_PER_BLOCK))?,
                    narrow(tokens)?,
                    narrow(spec.splits)?,
                ),
                block: (narrow(HEADS_PER_BLOCK * 32)?, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                &workspace.query,
                &*pages,
                block_tables,
                inputs.query_sequences,
                inputs.query_visible,
                &mut workspace.values,
                &mut workspace.statistics,
                narrow(tokens)?,
                narrow(spec.heads)?,
                narrow(spec.max_blocks)?,
                narrow(spec.block_size)?,
                layer.softmax_scale,
            ),
        )?;
        let rows = product(tokens, spec.heads)?;
        self.merge.launch(
            stream,
            LaunchConfig {
                grid: (narrow(rows)?, 1, 1),
                block: (128, 1, 1),
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
        absorption.expand_output(stream, layer.key_value_up, workspace, output)
    }

    fn validate(
        &self,
        tokens: usize,
        layer: &LatentAttentionLayer<'_>,
        cache: &LatentCacheView<'_>,
        inputs: &LatentAttentionInputs<'_>,
        output: &DeviceBuffer<bf16>,
    ) -> Result<()> {
        let spec = self.spec;
        if tokens == 0 || tokens > spec.max_queries || !layer.softmax_scale.is_finite() {
            return Err(Error::InvalidPagedKv("invalid latent attention launch"));
        }
        let rows = product(tokens, spec.heads)?;
        require(
            "latent query",
            product(rows, spec.nope_dim + spec.rope_dim)?,
            inputs.query.len(),
        )?;
        require("latent input", product(tokens, spec.row_width())?, inputs.latent.len())?;
        for (operand, buffer) in [
            ("latent positions", inputs.positions),
            ("latent slot mapping", inputs.slot_mapping),
            ("latent query sequences", inputs.query_sequences),
            ("latent query visibility", inputs.query_visible),
        ] {
            require(operand, tokens, buffer.len())?;
        }
        require("latent norm", spec.latent_rank, layer.latent_norm.len())?;
        let projected = product(spec.heads, spec.nope_dim + spec.value_dim)?;
        require(
            "latent up-projection",
            product(projected, spec.latent_rank)?,
            layer.key_value_up.len(),
        )?;
        require("latent rotary frequencies", spec.rope_dim / 2, layer.inverse_frequencies.len())?;
        require("latent block tables", spec.max_blocks, cache.block_tables.len())?;
        require("latent pages", self.spec.page_bytes(1)?, cache.pages.len())?;
        require("latent output", product(tokens, spec.heads * spec.value_dim)?, output.len())
    }
}
