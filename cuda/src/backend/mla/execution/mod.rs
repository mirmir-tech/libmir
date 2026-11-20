mod packed;
mod session;

use mircuda::{DeviceBuffer, bf16};

use super::{CudaLatentAttentionLayer, LatentGeometry, weights::QueryWeights};
use crate::{
    CudaBackend, DenseRole, Error, ExecutionPhase, PagedKvCache, Result,
    backend::{
        layer_scratch::SharedLayerScratch, linear::CheckpointProjection,
        state_space_hybrid::TailExecution,
    },
    kernels::{
        LatentAbsorption, LatentAttention, LatentAttentionInputs, LatentAttentionLayer,
        LatentAttentionWorkspace, LatentCacheView, ShiftedRmsNorm,
    },
};

/// Kernels, plans and scratch of one latent attention layer for a fixed
/// number of rows.
#[derive(Debug)]
pub struct CudaLatentAttentionExecution {
    backend: CudaBackend,
    geometry: LatentGeometry,
    layer: CudaLatentAttentionLayer,
    input_norm: ShiftedRmsNorm,
    query: Query,
    key_value_down: CheckpointProjection,
    output: CheckpointProjection,
    attention: LatentAttention,
    absorption: LatentAbsorption,
    workspace: LatentAttentionWorkspace,
    buffers: Buffers,
    tail: TailExecution,
    scratch: SharedLayerScratch,
    /// The sessions' shared pages, bound by the host-side packed preparation.
    pages: Option<PagedKvCache>,
    /// Device block table of a single-session pass, built on first use.
    session: Option<session::SessionPaging>,
}

#[derive(Debug)]
enum Query {
    Direct(Box<CheckpointProjection>),
    Latent(Box<LatentQuery>),
}

#[derive(Debug)]
struct LatentQuery {
    down: CheckpointProjection,
    norm: ShiftedRmsNorm,
    up: CheckpointProjection,
    /// `[tokens, query_rank]` before and after its norm.
    latent: [DeviceBuffer<bf16>; 2],
}

#[derive(Debug)]
struct Buffers {
    /// `[tokens, heads, nope + rope]`.
    query: DeviceBuffer<bf16>,
    /// `[tokens, latent + rope]`.
    latent: DeviceBuffer<bf16>,
    /// `[tokens, heads · value]`.
    attended: DeviceBuffer<bf16>,
}

/// Where the rows of one execution live in the cache.
struct Placement<'a> {
    positions: &'a DeviceBuffer<u32>,
    slot_mapping: &'a DeviceBuffer<u32>,
    sequences: &'a DeviceBuffer<u32>,
    tables: &'a DeviceBuffer<u32>,
    table_stride: usize,
}

impl CudaLatentAttentionExecution {
    pub(super) fn new(
        layer: &CudaLatentAttentionLayer,
        tokens: usize,
        phase: ExecutionPhase,
    ) -> Result<Self> {
        let backend = &layer.backend;
        let geometry = layer.geometry;
        let hidden = layer.norm.hidden_size;
        let projection = |input, output, role, weight| {
            CheckpointProjection::new(backend, tokens, input, output, role, weight)
        };
        let query = match &layer.weights.query {
            QueryWeights::Direct(weight) => Query::Direct(Box::new(projection(
                hidden,
                geometry.query_width(),
                DenseRole::AttentionQkv,
                weight,
            )?)),
            QueryWeights::Latent(weights) => {
                let (down, up) = (&weights.down, &weights.up);
                let rank = geometry
                    .query_rank
                    .ok_or(Error::InvalidDecoderKernel("query latent has no rank"))?;
                Query::Latent(Box::new(LatentQuery {
                    down: projection(hidden, rank, DenseRole::AttentionQkv, down)?,
                    norm: ShiftedRmsNorm::compile(
                        &backend.inner.compiler,
                        tokens,
                        rank,
                        geometry.latent_epsilon,
                        0.0,
                    )?,
                    up: projection(rank, geometry.query_width(), DenseRole::AttentionQkv, up)?,
                    latent: [allocate(backend, tokens * rank)?, allocate(backend, tokens * rank)?],
                }))
            },
        };
        let spec = geometry.kernel_spec(tokens, layer.block_size);
        let attention = LatentAttention::compile(&backend.inner.compiler, spec)?;
        let attended = geometry.heads * geometry.value_dim;
        Ok(Self {
            backend: backend.clone(),
            geometry,
            layer: layer.clone(),
            input_norm: ShiftedRmsNorm::compile(
                &backend.inner.compiler,
                tokens,
                hidden,
                layer.norm.rms_norm_epsilon,
                0.0,
            )?,
            query,
            key_value_down: projection(
                hidden,
                geometry.row_width(),
                DenseRole::AttentionQkv,
                &layer.weights.key_value_down,
            )?,
            output: projection(
                attended,
                hidden,
                DenseRole::AttentionOutput,
                &layer.weights.output,
            )?,
            workspace: attention.workspace(&backend.inner.pool, &backend.inner.stream)?,
            absorption: LatentAbsorption::new(
                &backend.inner.context,
                &backend.inner.stream,
                spec,
                tokens,
            )?,
            attention,
            buffers: Buffers {
                query: allocate(backend, tokens * geometry.query_width())?,
                latent: allocate(backend, tokens * geometry.row_width())?,
                attended: allocate(backend, tokens * attended)?,
            },
            tail: TailExecution::new(
                backend,
                layer.norm,
                layer.feed_forward.as_ref(),
                tokens,
                phase,
            )?,
            scratch: backend.inner.layer_scratch.acquire(backend, tokens, hidden)?,
            pages: None,
            session: None,
        })
    }

    // The scratch guard is used by the last statement; the lint cannot see it.
    #[allow(clippy::significant_drop_tightening)]
    fn run(
        &mut self,
        input: &DeviceBuffer<bf16>,
        pages: &mut PagedKvCache,
        placement: &Placement<'_>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let stream = self.backend.inner.stream.clone();
        let mut guard = self.scratch.lock()?;
        let scratch = &mut *guard;
        let weights = &self.layer.weights;
        self.input_norm.execute(
            &stream,
            input,
            bf16_buffer(&weights.input_norm)?,
            &mut scratch.normalized,
        )?;
        match &mut self.query {
            Query::Direct(projection) => {
                projection.execute(&scratch.normalized, &mut self.buffers.query)?;
            },
            Query::Latent(query) => {
                let LatentQuery { down, norm, up, latent } = &mut **query;
                let QueryWeights::Latent(query_weights) = &weights.query else {
                    return Err(Error::InvalidDecoderKernel("query latent weights are missing"));
                };
                let [raw, normalized] = latent;
                down.execute(&scratch.normalized, raw)?;
                norm.execute(&stream, raw, bf16_buffer(&query_weights.norm)?, normalized)?;
                up.execute(normalized, &mut self.buffers.query)?;
            },
        }
        self.key_value_down.execute(&scratch.normalized, &mut self.buffers.latent)?;
        self.attention.execute(
            &stream,
            &mut self.absorption,
            &LatentAttentionLayer {
                latent_norm: bf16_buffer(&weights.key_value_norm)?,
                key_value_up: bf16_buffer(&weights.key_value_up)?,
                inverse_frequencies: &self.layer.inverse_frequencies,
                epsilon: self.geometry.latent_epsilon,
                rope_scale: self.geometry.rope_scale,
                softmax_scale: self.geometry.softmax_scale,
            },
            LatentCacheView {
                pages: pages.latent_rows_mut()?,
                block_tables: placement.tables,
                table_stride: placement.table_stride,
            },
            LatentAttentionInputs {
                query: &self.buffers.query,
                latent: &self.buffers.latent,
                positions: placement.positions,
                slot_mapping: placement.slot_mapping,
                query_sequences: placement.sequences,
            },
            &mut self.workspace,
            &mut self.buffers.attended,
        )?;
        self.output.execute(&self.buffers.attended, &mut scratch.attention)?;
        self.tail.execute(input, scratch, output)
    }
}

fn allocate(backend: &CudaBackend, elements: usize) -> Result<DeviceBuffer<bf16>> {
    Ok(backend.inner.pool.allocate(&backend.inner.stream, elements)?)
}

fn bf16_buffer(tensor: &crate::CudaTensor) -> Result<&DeviceBuffer<bf16>> {
    tensor.as_bf16().ok_or_else(|| Error::DTypeMismatch {
        name: tensor.name().into(),
        expected: "BF16",
    })
}
