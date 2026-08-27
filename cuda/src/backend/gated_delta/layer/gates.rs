use mircuda::{DeviceBuffer, bf16};

use super::{
    AffineGatedDeltaLayerConfig, AffineGatedDeltaLayerWeights, CudaAffineGatedDeltaExecution,
};
use crate::{
    Bf16LinearPair, Bf16LinearPairWeights, CudaBackend, CudaTensor, Error, ExecutionPhase, Result,
    kernels::{GatedDeltaAlphaBeta, GatedDeltaAlphaBetaSplit},
};

const PAIRED_ALPHA_BETA_MIN_TOKENS: usize = 32;

#[derive(Debug)]
pub(super) enum DenseAlphaBeta {
    Direct(GatedDeltaAlphaBeta),
    Paired {
        projection: Bf16LinearPair,
        weights: Bf16LinearPairWeights,
        split: GatedDeltaAlphaBetaSplit,
    },
}

impl DenseAlphaBeta {
    pub(super) const fn paired(&self) -> bool {
        matches!(self, Self::Paired { .. })
    }
}

pub(super) fn prepare_dense_alpha_beta(
    backend: &CudaBackend,
    config: AffineGatedDeltaLayerConfig,
    weights: &AffineGatedDeltaLayerWeights,
    packed: Option<&Bf16LinearPairWeights>,
    tokens: usize,
) -> Result<Option<DenseAlphaBeta>> {
    weights
        .alpha
        .dense_bf16()
        .zip(weights.beta.dense_bf16())
        .map(|_| match packed.filter(|_| tokens >= PAIRED_ALPHA_BETA_MIN_TOKENS) {
            Some(weights) => Ok(DenseAlphaBeta::Paired {
                projection: Bf16LinearPair::new(
                    backend,
                    ExecutionPhase::Prefill,
                    tokens,
                    config.hidden_size,
                    config.value_heads,
                )?,
                weights: weights.clone(),
                split: GatedDeltaAlphaBetaSplit::compile(
                    &backend.inner.compiler,
                    tokens,
                    config.value_heads,
                )?,
            }),
            None => GatedDeltaAlphaBeta::compile(
                &backend.inner.compiler,
                tokens,
                config.hidden_size,
                config.value_heads,
            )
            .map(DenseAlphaBeta::Direct),
        })
        .transpose()
}

impl CudaAffineGatedDeltaExecution {
    pub(super) fn project_qkv_gate(&mut self, input: &DeviceBuffer<bf16>) -> Result<bool> {
        match (&mut self.packed_qkv_gate, &mut self.scratch.packed_qkv_gate) {
            (Some(projection), Some(packed)) => {
                projection.execute(input, packed)?;
                Ok(true)
            },
            (None, None) => {
                self.qkv
                    .as_mut()
                    .ok_or(Error::InvalidExecutionPlan("Gated Delta QKV projection is missing"))?
                    .execute(input, &mut self.scratch.mixed)?;
                self.gate
                    .as_mut()
                    .ok_or(Error::InvalidExecutionPlan("Gated Delta gate projection is missing"))?
                    .execute(input, &mut self.scratch.gate)?;
                Ok(false)
            },
            _ => Err(Error::InvalidExecutionPlan(
                "Gated Delta packed projection contract is incomplete",
            )),
        }
    }

    pub(super) fn project_alpha_beta(&mut self, input: &DeviceBuffer<bf16>) -> Result<()> {
        match &mut self.dense_alpha_beta {
            Some(DenseAlphaBeta::Direct(operation)) => operation.execute(
                &self.backend.inner.stream,
                input,
                bf16(self.weights.alpha.dense_bf16())?,
                bf16(self.weights.beta.dense_bf16())?,
                &mut self.scratch.alpha,
                &mut self.scratch.beta,
            ),
            Some(DenseAlphaBeta::Paired { projection, weights, split }) => {
                let packed = self.scratch.packed_alpha_beta.as_mut().ok_or(
                    Error::InvalidExecutionPlan("packed Gated Delta alpha/beta output is missing"),
                )?;
                projection.execute(input, weights, packed)?;
                split.execute(
                    &self.backend.inner.stream,
                    packed,
                    &mut self.scratch.alpha,
                    &mut self.scratch.beta,
                )
            },
            None => {
                self.alpha
                    .as_mut()
                    .ok_or(Error::InvalidExecutionPlan("Gated Delta alpha projection is missing"))?
                    .execute(input, &mut self.scratch.alpha)?;
                self.beta
                    .as_mut()
                    .ok_or(Error::InvalidExecutionPlan("Gated Delta beta projection is missing"))?
                    .execute(input, &mut self.scratch.beta)
            },
        }
    }
}

fn bf16(tensor: Option<&CudaTensor>) -> Result<&DeviceBuffer<bf16>> {
    tensor
        .and_then(CudaTensor::as_bf16)
        .ok_or(Error::InvalidExecutionPlan("dense Gated Delta gate weight is missing"))
}
