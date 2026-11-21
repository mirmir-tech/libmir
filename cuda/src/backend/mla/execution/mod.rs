mod packed;
mod session;

use mircuda::{DeviceBuffer, bf16};

use super::{
    CudaLatentAttentionLayer, LatentGeometry, LatentScratch, LatentScratchKey,
    weights::QueryWeights,
};
use crate::{
    CudaBackend, DenseRole, Error, ExecutionPhase, PagedKvCache, Result,
    backend::{
        layer_scratch::SharedLayerScratch, linear::CheckpointProjection,
        scratch_pool::SharedScratch, state_space_hybrid::TailExecution,
    },
    kernels::{
        LatentAbsorption, LatentAttention, LatentAttentionInputs, LatentAttentionLayer,
        LatentCacheView, ShiftedRmsNorm,
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
    /// Buffers shared by every latent layer of this shape.
    latent_scratch: SharedScratch<LatentScratch>,
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
                }))
            },
        };
        let spec = geometry.kernel_spec(tokens, layer.block_size);
        let attention = LatentAttention::compile(&backend.inner.compiler, spec)?;
        let attended = geometry.heads * geometry.value_dim;
        let key = LatentScratchKey::new(&geometry, &attention, tokens);
        let latent_scratch = backend
            .inner
            .latent_attention_scratch
            .acquire(key.clone(), || LatentScratch::new(backend, &key, &attention))?;
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
            latent_scratch,
            absorption: LatentAbsorption::new(
                &backend.inner.context,
                &backend.inner.stream,
                spec,
                layer.weights.up.weights().layout(),
                tokens,
            )?,
            attention,
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
        let latent_handle = self.latent_scratch.clone_handle();
        let mut latent_guard = latent_handle.lock()?;
        let LatentScratch {
            workspace,
            query: query_rows,
            latent,
            attended,
            query_latent,
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
        self.attention.execute(
            &stream,
            &mut self.absorption,
            &LatentAttentionLayer {
                latent_norm: bf16_buffer(&weights.key_value_norm)?,
                up: weights.up.weights(),
                inverse_frequencies: &self.layer.inverse_frequencies,
                epsilon: self.geometry.latent_epsilon,
                rope_scale: self.geometry.rope_scale,
                softmax_scale: self.geometry.softmax_scale,
                temperature: self.geometry.temperature,
            },
            LatentCacheView {
                pages: pages.latent_rows_mut()?,
                block_tables: placement.tables,
                table_stride: placement.table_stride,
            },
            LatentAttentionInputs {
                query: query_rows,
                latent,
                positions: placement.positions,
                slot_mapping: placement.slot_mapping,
                query_sequences: placement.sequences,
            },
            workspace,
            attended,
        )?;
        self.output.execute(attended, &mut scratch.attention)?;
        self.tail.execute(input, scratch, output)
    }
}

fn bf16_buffer(tensor: &crate::CudaTensor) -> Result<&DeviceBuffer<bf16>> {
    tensor.as_bf16().ok_or_else(|| Error::DTypeMismatch {
        name: tensor.name().into(),
        expected: "BF16",
    })
}
