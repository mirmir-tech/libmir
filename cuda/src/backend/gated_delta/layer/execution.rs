use mircuda::{DeviceBuffer, bf16};

use super::{
    AffineGatedDeltaLayerConfig, AffineGatedDeltaLayerWeights, CudaGatedDeltaState,
    gates::prepare_dense_alpha_beta, require_exact, scratch::GatedDeltaScratch,
};
use crate::{
    CudaBackend, DenseRole, Error, GatedDeltaInputs, Result,
    backend::{
        gated_delta::CudaGatedDeltaBatchState,
        linear::{CheckpointProjection, CheckpointProjectionWeight},
    },
    kernels::{GatedDeltaAlphaBeta, GatedDeltaTransformSpec, GatedDeltaTransforms},
};

#[derive(Debug)]
pub struct CudaAffineGatedDeltaExecution {
    pub(super) backend: CudaBackend,
    pub(super) config: AffineGatedDeltaLayerConfig,
    pub(super) tokens: usize,
    qkv: Option<CheckpointProjection>,
    gate: Option<CheckpointProjection>,
    packed_qkv_gate: Option<CheckpointProjection>,
    pub(super) alpha: Option<CheckpointProjection>,
    pub(super) beta: Option<CheckpointProjection>,
    pub(super) dense_alpha_beta: Option<GatedDeltaAlphaBeta>,
    pub(super) output: CheckpointProjection,
    pub(super) transforms: GatedDeltaTransforms,
    pub(super) weights: AffineGatedDeltaLayerWeights,
    pub(super) scratch: GatedDeltaScratch,
    pub(super) batch_state: Option<CudaGatedDeltaBatchState>,
}

impl CudaAffineGatedDeltaExecution {
    pub(super) fn new(
        backend: &CudaBackend,
        config: AffineGatedDeltaLayerConfig,
        weights: &AffineGatedDeltaLayerWeights,
        packed_qkv_gate: Option<&CheckpointProjectionWeight>,
        tokens: usize,
    ) -> Result<Self> {
        let projection = |input, output, weights| {
            CheckpointProjection::new(
                backend,
                tokens,
                input,
                output,
                DenseRole::AttentionQkv,
                weights,
            )
        };
        let value = config.value_width()?;
        let mixed = config.mixed_width()?;
        let packed_output = mixed
            .checked_add(value)
            .ok_or(Error::InvalidDecoderKernel("packed Gated Delta size overflow"))?;
        let packed = packed_qkv_gate
            .map(|weight| {
                CheckpointProjection::new(
                    backend,
                    tokens,
                    config.hidden_size,
                    packed_output,
                    DenseRole::AttentionQkv,
                    weight,
                )
            })
            .transpose()?;
        let dense_alpha_beta = prepare_dense_alpha_beta(backend, config, weights, tokens)?;
        Ok(Self {
            backend: backend.clone(),
            config,
            tokens,
            qkv: if packed.is_none() {
                Some(projection(config.hidden_size, mixed, &weights.qkv)?)
            } else {
                None
            },
            gate: if packed.is_none() {
                Some(projection(config.hidden_size, value, &weights.gate)?)
            } else {
                None
            },
            packed_qkv_gate: packed,
            alpha: dense_alpha_beta
                .is_none()
                .then(|| projection(config.hidden_size, config.value_heads, &weights.alpha))
                .transpose()?,
            beta: dense_alpha_beta
                .is_none()
                .then(|| projection(config.hidden_size, config.value_heads, &weights.beta))
                .transpose()?,
            dense_alpha_beta,
            output: CheckpointProjection::new(
                backend,
                tokens,
                value,
                config.hidden_size,
                DenseRole::AttentionOutput,
                &weights.output,
            )?,
            transforms: GatedDeltaTransforms::compile(
                &backend.inner.compiler,
                GatedDeltaTransformSpec {
                    tokens,
                    key_heads: config.key_heads,
                    value_heads: config.value_heads,
                    key_dim: config.key_dim,
                    value_dim: config.value_dim,
                    epsilon: config.rms_norm_epsilon,
                    norm_weight_shift: config.norm_weight_shift,
                },
            )?,
            weights: weights.clone(),
            scratch: GatedDeltaScratch::new(backend, config, tokens, packed_qkv_gate.is_some())?,
            batch_state: None,
        })
    }

    pub fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        state: &mut CudaGatedDeltaState,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.validate(input, state, output)?;
        let packed = self.project_qkv_gate(input)?;
        self.project_alpha_beta(input)?;
        let stream = &self.backend.inner.stream;
        if packed {
            let source = self
                .scratch
                .packed_qkv_gate
                .as_ref()
                .ok_or(Error::InvalidExecutionPlan("packed Gated Delta output is missing"))?;
            state.convolve_silu_strided(
                self.tokens,
                source,
                bf16(&self.weights.convolution)?,
                &mut self.scratch.convolved,
                self.config.mixed_width()? + self.config.value_width()?,
                0,
            )?;
        } else {
            state.convolve_silu(
                self.tokens,
                &self.scratch.mixed,
                bf16(&self.weights.convolution)?,
                &mut self.scratch.convolved,
            )?;
        }
        self.transforms.split_normalize(
            stream,
            &self.scratch.convolved,
            &mut self.scratch.normalized_query,
            &mut self.scratch.normalized_key,
            &mut self.scratch.value,
        )?;
        state.execute(
            self.tokens,
            GatedDeltaInputs {
                query: &self.scratch.normalized_query,
                key: &self.scratch.normalized_key,
                value: &self.scratch.value,
                alpha: &self.scratch.alpha,
                beta: &self.scratch.beta,
                a_log: bf16(&self.weights.a_log)?,
                dt_bias: bf16(&self.weights.dt_bias)?,
            },
            &mut self.scratch.recurrent,
        )?;
        if packed {
            self.transforms.norm_gate_strided(
                stream,
                &self.scratch.recurrent,
                self.scratch
                    .packed_qkv_gate
                    .as_ref()
                    .ok_or(Error::InvalidExecutionPlan("packed Gated Delta output is missing"))?,
                bf16(&self.weights.norm)?,
                &mut self.scratch.gated,
                self.config.mixed_width()? + self.config.value_width()?,
                self.config.mixed_width()?,
            )?;
        } else {
            self.transforms.norm_gate(
                stream,
                &self.scratch.recurrent,
                &self.scratch.gate,
                bf16(&self.weights.norm)?,
                &mut self.scratch.gated,
            )?;
        }
        self.output.execute(&self.scratch.gated, output)
    }

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

    pub(super) fn validate(
        &self,
        input: &DeviceBuffer<bf16>,
        state: &CudaGatedDeltaState,
        output: &DeviceBuffer<bf16>,
    ) -> Result<()> {
        if state.config() != self.config.state()? {
            return Err(Error::InvalidDecoderKernel("Gated Delta state config mismatch"));
        }
        require_exact(
            "affine Gated Delta input",
            self.tokens * self.config.hidden_size,
            input.len(),
        )?;
        require_exact(
            "affine Gated Delta output",
            self.tokens * self.config.hidden_size,
            output.len(),
        )
    }
}

fn bf16(tensor: &crate::CudaTensor) -> Result<&DeviceBuffer<bf16>> {
    tensor.as_bf16().ok_or_else(|| Error::DTypeMismatch {
        name: tensor.name().into(),
        expected: "BF16",
    })
}
