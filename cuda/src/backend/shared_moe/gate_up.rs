use crate::{
    CudaBackend, DenseRole, Result,
    backend::{
        linear::{CheckpointProjection, CheckpointProjectionWeight, MarlinNvFp4Bf16Linear},
        shared_moe::{AffineSharedExpertMoeConfig, AffineSharedExpertMoeWeights},
    },
};

#[derive(Debug)]
pub(super) enum SharedGateUp {
    Separate {
        gate: Box<CheckpointProjection>,
        up: Box<CheckpointProjection>,
    },
    PackedNvFp4(MarlinNvFp4Bf16Linear),
}

pub(super) fn prepare_shared_gate_up(
    backend: &CudaBackend,
    config: AffineSharedExpertMoeConfig,
    weights: &AffineSharedExpertMoeWeights,
    tokens: usize,
) -> Result<SharedGateUp> {
    if let (
        CheckpointProjectionWeight::NvFp4WeightOnly(gate),
        CheckpointProjectionWeight::NvFp4WeightOnly(up),
    ) = (&weights.shared_gate, &weights.shared_up)
        && let Some(operation) = MarlinNvFp4Bf16Linear::new_pair(backend, tokens, gate, up)?
    {
        return Ok(SharedGateUp::PackedNvFp4(operation));
    }
    Ok(SharedGateUp::Separate {
        gate: Box::new(CheckpointProjection::new(
            backend,
            tokens,
            config.hidden_size,
            config.shared_intermediate_size,
            DenseRole::DenseGateUp,
            &weights.shared_gate,
        )?),
        up: Box::new(CheckpointProjection::new(
            backend,
            tokens,
            config.hidden_size,
            config.shared_intermediate_size,
            DenseRole::DenseGateUp,
            &weights.shared_up,
        )?),
    })
}
