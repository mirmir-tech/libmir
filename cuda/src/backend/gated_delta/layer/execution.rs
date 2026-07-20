use mircuda::{DeviceBuffer, bf16};

use super::{
    AffineGatedDeltaLayerConfig, AffineGatedDeltaLayerWeights, CudaGatedDeltaState,
    scratch::GatedDeltaScratch,
};
use crate::{
    CudaBackend, Error, GatedDeltaInputs, Result,
    backend::linear::AffineProjection,
    kernels::{GatedDeltaTransformSpec, GatedDeltaTransforms},
};

#[derive(Debug)]
pub struct CudaAffineGatedDeltaExecution {
    backend: CudaBackend,
    config: AffineGatedDeltaLayerConfig,
    tokens: usize,
    qkv: AffineProjection,
    gate: AffineProjection,
    alpha: AffineProjection,
    beta: AffineProjection,
    output: AffineProjection,
    transforms: GatedDeltaTransforms,
    weights: AffineGatedDeltaLayerWeights,
    scratch: GatedDeltaScratch,
}

impl CudaAffineGatedDeltaExecution {
    pub(super) fn new(
        backend: &CudaBackend,
        config: AffineGatedDeltaLayerConfig,
        weights: &AffineGatedDeltaLayerWeights,
        tokens: usize,
    ) -> Result<Self> {
        let projection = |input, output, weights| {
            AffineProjection::new(
                backend,
                tokens,
                input,
                output,
                config.group_size,
                config.bits,
                weights,
            )
        };
        let value = config.value_width()?;
        Ok(Self {
            backend: backend.clone(),
            config,
            tokens,
            qkv: projection(config.hidden_size, config.mixed_width()?, &weights.qkv)?,
            gate: projection(config.hidden_size, value, &weights.gate)?,
            alpha: projection(config.hidden_size, config.value_heads, &weights.alpha)?,
            beta: projection(config.hidden_size, config.value_heads, &weights.beta)?,
            output: projection(value, config.hidden_size, &weights.output)?,
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
            scratch: GatedDeltaScratch::new(backend, config, tokens)?,
        })
    }

    pub fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        state: &mut CudaGatedDeltaState,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.validate(input, state, output)?;
        let stream = &self.backend.inner.stream;
        self.qkv.execute(input, &mut self.scratch.mixed)?;
        state.convolve_silu(
            self.tokens,
            &self.scratch.mixed,
            bf16(&self.weights.convolution)?,
            &mut self.scratch.convolved,
        )?;
        self.transforms.split(
            stream,
            &self.scratch.convolved,
            &mut self.scratch.query,
            &mut self.scratch.key,
            &mut self.scratch.value,
        )?;
        self.transforms.normalize_qk(
            stream,
            &self.scratch.query,
            &self.scratch.key,
            &mut self.scratch.normalized_query,
            &mut self.scratch.normalized_key,
        )?;
        self.gate.execute(input, &mut self.scratch.gate)?;
        self.alpha.execute(input, &mut self.scratch.alpha)?;
        self.beta.execute(input, &mut self.scratch.beta)?;
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
        self.transforms.norm_gate(
            stream,
            &self.scratch.recurrent,
            &self.scratch.gate,
            bf16(&self.weights.norm)?,
            &mut self.scratch.gated,
        )?;
        self.output.execute(&self.scratch.gated, output)
    }

    fn validate(
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

fn require_exact(name: &str, expected: usize, actual: usize) -> Result<()> {
    if expected != actual {
        return Err(Error::InvalidTensorSize { name: name.into(), expected, actual });
    }
    Ok(())
}
