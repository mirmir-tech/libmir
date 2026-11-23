mod packed;
mod run;
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
        LatentAbsorption, LatentAttention, LatentExpandedRows, LatentExpansion, ShiftedRmsNorm,
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
    /// Expanded prefill attention, for prefill plans of enough rows.
    expansion: Option<LatentExpansion>,
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
    /// The sequences of a prefill pass, when the pass may attend expanded.
    expanded: Option<LatentExpandedRows<'a>>,
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
        let expansion =
            expansion(backend, &attention, layer.weights.up.weights().layout(), tokens, phase)?;
        let key = LatentScratchKey::new(&geometry, &attention, tokens, expansion.as_ref());
        let latent_scratch = backend.inner.latent_attention_scratch.acquire(key.clone(), || {
            LatentScratch::new(backend, &key, &attention, expansion.as_ref())
        })?;
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
            expansion,
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
}

/// Prefill plans of at least this many rows may attend expanded.
const EXPANDED_MIN_TOKENS: usize = 16;
/// Pool bytes of the expanded keys, values and context latents.
const EXPANSION_BYTES: usize = 512 << 20;

fn expansion(
    backend: &CudaBackend,
    attention: &LatentAttention,
    layout: crate::kernels::LatentUpLayout,
    tokens: usize,
    phase: ExecutionPhase,
) -> Result<Option<LatentExpansion>> {
    if phase != ExecutionPhase::Prefill || tokens < EXPANDED_MIN_TOKENS {
        return Ok(None);
    }
    let spec = attention.spec();
    let capacity = EXPANSION_BYTES / LatentExpansion::buffer_bytes(spec, 1, (0, 0)).max(1);
    LatentExpansion::new(&backend.inner.context, &backend.inner.stream, spec, layout, capacity)
}

fn bf16_buffer(tensor: &crate::CudaTensor) -> Result<&DeviceBuffer<bf16>> {
    tensor.as_bf16().ok_or_else(|| Error::DTypeMismatch {
        name: tensor.name().into(),
        expected: "BF16",
    })
}
