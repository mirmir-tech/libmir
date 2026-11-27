//! Decode attention through split-KV `FlashAttention` 2, rescaled by the
//! learned sinks from its log-sum-exp (`sink_scale`): one query row per
//! sequence over the paged cache, or over a windowed layer's ring, whose
//! overwritten slots all lie before the window the mask keeps.

use mircuda::{DeviceBuffer, FmhaCausalWindow, LaunchConfig, Stream, bf16};

use super::{ClampedRoutedAttention, narrow};
use crate::{CudaBackend, PagedPrefillBatch, Result};

const MAX_SPLITS: usize = 16;

/// Split partials and log-sum-exps of decode `FlashAttention`, shared by
/// the layers of one plan (they run in order on one stream).
#[derive(Debug)]
pub struct ClampedRoutedFmhaWorkspace {
    values: DeviceBuffer<f32>,
    statistics: DeviceBuffer<f32>,
    /// `f32` log-sum-exps, stored as twice as many BF16 slots.
    lse: DeviceBuffer<bf16>,
    rows: usize,
    multiprocessors: usize,
}

impl ClampedRoutedFmhaWorkspace {
    pub(crate) fn new(
        backend: &CudaBackend,
        rows: usize,
        query_heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        let statistics = MAX_SPLITS * rows * query_heads;
        Ok(Self {
            values: backend.pool().allocate(backend.stream(), statistics * head_dim)?,
            statistics: backend.pool().allocate(backend.stream(), statistics)?,
            lse: backend.pool().allocate(backend.stream(), 2 * rows * query_heads)?,
            rows,
            multiprocessors: usize::try_from(
                backend.execution_planner().hardware().multiprocessor_count().get(),
            )?,
        })
    }
}

impl ClampedRoutedFmhaWorkspace {
    /// Whether a decode batch of `rows` sequences fits.
    #[must_use]
    pub(crate) const fn fits(&self, rows: usize) -> bool {
        rows <= self.rows
    }
}

impl ClampedRoutedAttention {
    #[must_use]
    pub(crate) const fn has_fmha(&self) -> bool {
        self.fmha.is_some()
    }

    /// Attention of a decode batch, `false` when this layer has no
    /// `FlashAttention` plan or the batch does not fit the workspace.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_fmha_decode(
        &self,
        stream: &Stream,
        query: &DeviceBuffer<bf16>,
        key_pages: &DeviceBuffer<u8>,
        value_pages: &DeviceBuffer<u8>,
        batch: &PagedPrefillBatch,
        tables: &DeviceBuffer<u32>,
        sinks: &DeviceBuffer<bf16>,
        workspace: &mut ClampedRoutedFmhaWorkspace,
        output: &mut DeviceBuffer<bf16>,
        scale: f32,
    ) -> Result<bool> {
        let Some(fmha) = &self.fmha else {
            return Ok(false);
        };
        let rows = batch.active();
        if batch.max_query_tokens() != 1 || rows > workspace.rows {
            return Ok(false);
        }
        let context = batch.fmha_max_context_tokens();
        let splits = self.splits(fmha.window, context, rows, workspace.multiprocessors);
        let args = (rows, rows, 1, context, batch.max_blocks(), batch.cache_config().block_size);
        if splits == 1 {
            fmha.plan.execute_paged_varlen_windowed(
                stream,
                query,
                key_pages,
                value_pages,
                output,
                batch.query_starts(),
                batch.token_counts(),
                batch.context_starts(),
                tables,
                &mut workspace.lse,
                args.0,
                args.1,
                args.2,
                args.3,
                args.4,
                args.5,
                fmha.window,
                scale,
            )?;
        } else {
            fmha.plan.execute_paged_varlen_split(
                stream,
                query,
                key_pages,
                value_pages,
                output,
                batch.query_starts(),
                batch.token_counts(),
                batch.context_starts(),
                tables,
                &mut workspace.lse,
                &mut workspace.values,
                &mut workspace.statistics,
                splits,
                args.0,
                args.1,
                args.2,
                args.3,
                args.4,
                args.5,
                fmha.window,
                scale,
            )?;
        }
        self.sink_scale.launch(
            stream,
            LaunchConfig::for_elements(rows * self.query_heads * self.head_dim, 256)?,
            (
                output,
                &workspace.lse,
                sinks,
                narrow(rows)?,
                narrow(self.query_heads)?,
                narrow(self.head_dim)?,
            ),
        )?;
        Ok(true)
    }

    /// Enough KV splits to fill the device twice over, at most one per
    /// `FlashAttention` key tile of the visible context. Decode, full or
    /// windowed, shares each K/V head across its query group (one block per
    /// K/V head).
    fn splits(
        &self,
        window: FmhaCausalWindow,
        context: usize,
        rows: usize,
        multiprocessors: usize,
    ) -> usize {
        let visible = match window {
            FmhaCausalWindow::Full => context,
            FmhaCausalWindow::Sliding(tokens) => tokens.get().min(context),
        };
        // `run_mha_fwd_splitkv_dispatch` key tile: 256 keys for head_dim 64,
        // 128 up to 128, 64 beyond.
        let tile = match self.head_dim {
            0..=64 => 256,
            65..=128 => 128,
            _ => 64,
        };
        let blocks = (rows * self.kv_heads).max(1);
        visible
            .div_ceil(tile)
            .min((2 * multiprocessors).div_ceil(blocks))
            .clamp(1, MAX_SPLITS)
    }
}
