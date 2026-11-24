//! Expanded latent attention for prefill.
//!
//! The visible context's latent rows expand into per-head keys and values
//! (`latent · W_UKᵀ`, `latent · W_UVᵀ`) and one fused attention pass over
//! them replaces the absorbed latent attention. Absorbed attention works in
//! the latent width (`latent_rank + rope_dim` per score, `latent_rank` per
//! value) and pays it for every query; expanding once per chunk is cheaper
//! when a chunk holds many queries.

mod context;
mod execute;

use std::collections::HashMap;

use mircuda::{
    Context, CublasGemmOperand, CublasGemmPlan, CublasGemmSpec, DeviceBuffer, FmhaBf16Plan,
    FmhaBf16Spec, MemoryPool, Stream, bf16,
};

use super::{LatentAttentionSpec, LatentUpLayout};
use crate::{Error, Result, kernels::geometry::product};

/// Context rows the expansion plans round up to, and the page size the
/// fused attention reads the expanded keys and values in.
const CONTEXT_ROWS: usize = 64;

/// Plans of expanded attention for one layer geometry.
#[derive(Debug)]
pub struct LatentExpansion {
    spec: LatentAttentionSpec,
    layout: LatentUpLayout,
    /// The fused attention's head width: the query/key width rounded up.
    head_dim: usize,
    /// Context tokens the buffers hold.
    capacity: usize,
    attention: FmhaBf16Plan,
    /// Key and value up-projection plans by padded context rows.
    plans: HashMap<usize, [CublasGemmPlan<bf16, bf16>; 2]>,
    context: Context,
}

/// Scratch of expanded attention, shared by every layer of one geometry.
#[derive(Debug)]
pub struct LatentExpansionBuffers {
    /// `[capacity, latent_rank + rope_dim]` cache rows of the context.
    latent: DeviceBuffer<bf16>,
    /// `[capacity, heads, head_dim]`.
    keys: DeviceBuffer<bf16>,
    values: DeviceBuffer<bf16>,
    /// `[queries, heads, head_dim]`.
    queries: DeviceBuffer<bf16>,
    /// `[queries, heads, head_dim]` when values are narrower than heads.
    output: Option<DeviceBuffer<bf16>>,
    /// Query starts, context length and context starts of one sequence.
    starts: DeviceBuffer<u32>,
    /// `[sequences + 1]` first key page of every sequence.
    regions: DeviceBuffer<u32>,
    /// `[sequences, capacity / CONTEXT_ROWS]` key pages of every sequence.
    table: DeviceBuffer<u32>,
    /// Sequences one launch holds.
    sequences: usize,
    /// `[queries, heads]` softmax statistics.
    statistics: DeviceBuffer<f32>,
}

impl LatentExpansion {
    /// Plans for contexts of up to `capacity` tokens; `None` when no fused
    /// attention width holds the query/key and value widths.
    pub fn new(
        context: &Context,
        stream: &Stream,
        spec: LatentAttentionSpec,
        layout: LatentUpLayout,
        capacity: usize,
    ) -> Result<Option<Self>> {
        let Some(head_dim) = head_dim(spec) else {
            return Ok(None);
        };
        if capacity == 0 {
            return Ok(None);
        }
        let attention = FmhaBf16Spec::new(spec.heads, spec.heads, head_dim, head_dim)?;
        Ok(Some(Self {
            spec,
            layout,
            head_dim,
            capacity: capacity.div_ceil(CONTEXT_ROWS) * CONTEXT_ROWS,
            attention: FmhaBf16Plan::new(context, stream, attention)?,
            plans: HashMap::new(),
            context: context.clone(),
        }))
    }

    #[must_use]
    /// Bytes of [`LatentExpansionBuffers`] for `capacity` context tokens,
    /// `queries` query tokens and `sequences` sequences.
    pub fn buffer_bytes(
        spec: LatentAttentionSpec,
        capacity: usize,
        (queries, sequences): (usize, usize),
    ) -> usize {
        let head_rows = spec.heads * head_dim(spec).unwrap_or(0);
        let pages = capacity.div_ceil(CONTEXT_ROWS);
        2 * (capacity * (spec.row_width() + 2 * head_rows) + 2 * queries * head_rows)
            + 4 * (6 + sequences * (pages + 1) + queries * spec.heads)
    }

    /// Context tokens one expansion holds.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Whether one launch of `rows` fits these plans and `buffers`.
    #[must_use]
    pub fn admits(&self, buffers: &LatentExpansionBuffers, rows: &[LatentExpandedRow]) -> bool {
        let pages: usize = rows.iter().map(|row| row.context().div_ceil(CONTEXT_ROWS)).sum();
        !rows.is_empty() && rows.len() <= buffers.sequences && pages * CONTEXT_ROWS <= self.capacity
    }

    /// Scratch for launches of up to `queries` query tokens in up to
    /// `sequences` sequences.
    pub fn buffers(
        &self,
        pool: &MemoryPool,
        stream: &Stream,
        (queries, sequences): (usize, usize),
    ) -> Result<LatentExpansionBuffers> {
        let spec = self.spec;
        let rows = product(spec.heads, self.head_dim)?;
        let allocate = |elements| pool.allocate::<bf16>(stream, elements);
        let pages = self.capacity / CONTEXT_ROWS;
        Ok(LatentExpansionBuffers {
            latent: allocate(product(self.capacity, spec.row_width())?)?,
            keys: allocate(product(self.capacity, rows)?)?,
            values: allocate(product(self.capacity, rows)?)?,
            queries: allocate(product(queries, rows)?)?,
            output: if spec.value_dim < self.head_dim {
                Some(allocate(product(queries, rows)?)?)
            } else {
                None
            },
            starts: pool.allocate(stream, 5)?,
            regions: pool.allocate(stream, sequences + 1)?,
            table: pool.allocate(stream, product(sequences, pages)?)?,
            sequences,
            statistics: pool.allocate(stream, product(queries, spec.heads)?)?,
        })
    }

    /// Key and value up-projections of `rows` context rows.
    fn plans(
        &mut self,
        stream: &Stream,
        rows: usize,
    ) -> Result<&mut [CublasGemmPlan<bf16, bf16>; 2]> {
        if !self.plans.contains_key(&rows) {
            let spec = self.spec;
            let (rank, nope, value) = (spec.latent_rank, spec.nope_dim, spec.value_dim);
            // Every head reads the same latent rows (batch stride 0).
            let latent = operand(spec.row_width(), 0, false);
            let output = operand(product(spec.heads, self.head_dim)?, self.head_dim, false);
            let (key, value_weights) = match self.layout {
                LatentUpLayout::Fused => {
                    let head = product(nope + value, rank)?;
                    (operand(rank, head, true), operand(rank, head, true))
                },
                LatentUpLayout::Split => (
                    operand(nope, product(rank, nope)?, false),
                    operand(rank, product(value, rank)?, true),
                ),
            };
            let plan = |n, right| {
                CublasGemmPlan::new(
                    &self.context,
                    stream,
                    CublasGemmSpec::new((rows, n, rank, spec.heads), latent, right, output)?,
                )
            };
            let plans = [plan(nope, key)?, plan(value, value_weights)?];
            self.plans.insert(rows, plans);
        }
        self.plans
            .get_mut(&rows)
            .ok_or(Error::InvalidPagedKv("latent expansion plan is missing"))
    }
}

/// The fused attention width for the query/key and value widths: paged
/// `FlashAttention` 2 runs 64-, 128- and 256-wide heads (the 192-wide
/// `DeepSeek` keys pad to 256); wider geometries keep absorbed attention.
fn head_dim(spec: LatentAttentionSpec) -> Option<usize> {
    let width = (spec.nope_dim + spec.rope_dim).max(spec.value_dim);
    [64, 128, 256].into_iter().find(|&dim| dim >= width)
}

const fn operand(leading: usize, stride: usize, transposed: bool) -> CublasGemmOperand {
    CublasGemmOperand { leading, stride, transposed }
}

pub use execute::{LatentExpandedMetadata, LatentExpandedRow, LatentExpandedRows};
