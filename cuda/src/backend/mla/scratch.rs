//! Buffers every latent attention layer of one shape uses in turn: layers
//! run one after another on the runtime stream, so a 2,048-token prefill of a
//! 128-head, 60-layer model needs one set of about 1.3 GB, not sixty.

use mircuda::{DeviceBuffer, bf16};

use super::LatentGeometry;
use crate::{
    CudaBackend, Result,
    kernels::{LatentAttention, LatentAttentionWorkspace, LatentExpansion, LatentExpansionBuffers},
};

/// The shape that sizes [`LatentScratch`].
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(in crate::backend) struct LatentScratchKey {
    tokens: usize,
    heads: usize,
    query_width: usize,
    row_width: usize,
    latent_rank: usize,
    attended: usize,
    query_rank: Option<usize>,
    splits: usize,
    /// Context tokens the expanded buffers hold; zero without expansion.
    expansion_capacity: usize,
}

impl LatentScratchKey {
    pub(super) fn new(
        geometry: &LatentGeometry,
        attention: &LatentAttention,
        tokens: usize,
        expansion: Option<&LatentExpansion>,
    ) -> Self {
        Self {
            tokens,
            heads: geometry.heads,
            query_width: geometry.query_width(),
            row_width: geometry.row_width(),
            latent_rank: geometry.latent_rank,
            attended: geometry.heads * geometry.value_dim,
            query_rank: geometry.query_rank,
            splits: attention.spec().splits,
            expansion_capacity: expansion.map_or(0, LatentExpansion::capacity),
        }
    }
}

#[derive(Debug)]
pub(in crate::backend) struct LatentScratch {
    pub(super) workspace: LatentAttentionWorkspace,
    /// `[tokens, heads, nope + rope]`.
    pub(super) query: DeviceBuffer<bf16>,
    /// `[tokens, latent + rope]`.
    pub(super) latent: DeviceBuffer<bf16>,
    /// `[tokens, heads · value]`.
    pub(super) attended: DeviceBuffer<bf16>,
    /// `[tokens, query_rank]` before and after its norm.
    pub(super) query_latent: Option<[DeviceBuffer<bf16>; 2]>,
    /// Expanded keys, values and queries of prefill.
    pub(super) expansion: Option<LatentExpansionBuffers>,
}

/// Sequences one expanded prefill pass holds at most.
const EXPANDED_SEQUENCES: usize = 256;

impl LatentScratch {
    pub(super) fn new(
        backend: &CudaBackend,
        key: &LatentScratchKey,
        attention: &LatentAttention,
        expansion: Option<&LatentExpansion>,
    ) -> Result<Self> {
        let allocate = |elements: usize| -> Result<DeviceBuffer<bf16>> {
            Ok(backend.inner.pool.allocate(&backend.inner.stream, elements)?)
        };
        let tokens = key.tokens;
        Ok(Self {
            workspace: attention.workspace(&backend.inner.pool, &backend.inner.stream)?,
            query: allocate(tokens * key.query_width)?,
            latent: allocate(tokens * key.row_width)?,
            attended: allocate(tokens * key.attended)?,
            query_latent: key
                .query_rank
                .map(|rank| {
                    Ok::<_, crate::Error>([allocate(tokens * rank)?, allocate(tokens * rank)?])
                })
                .transpose()?,
            expansion: expansion
                .map(|expansion| {
                    let sequences = tokens.min(EXPANDED_SEQUENCES);
                    expansion.buffers(
                        &backend.inner.pool,
                        &backend.inner.stream,
                        (tokens, sequences),
                    )
                })
                .transpose()?,
        })
    }
}
