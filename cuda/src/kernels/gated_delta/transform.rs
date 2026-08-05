use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::super::geometry::{narrow, product, require};
use crate::{Error, Result};

cuda_export!(
    SplitNormalizeKernel = "libmir_cuda_gated_delta_split_normalize_bf16"(
        input: &DeviceBuffer<bf16>,
        normalized_query: &mut DeviceBuffer<bf16>,
        normalized_key: &mut DeviceBuffer<bf16>, value: &mut DeviceBuffer<bf16>,
        tokens: u32, key_heads: u32, value_heads: u32,
        key_dim: u32, value_dim: u32, epsilon: f32,
    )
);

cuda_export!(
    NormGateKernel = "libmir_cuda_gated_delta_norm_gate_bf16"(
        input: &DeviceBuffer<bf16>, gate: &DeviceBuffer<bf16>,
        weight: &DeviceBuffer<bf16>, output: &mut DeviceBuffer<bf16>,
        rows: u32, columns: u32, epsilon: f32, weight_shift: f32,
        value_heads: u32, gate_stride: u32, gate_offset: u32,
    )
);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GatedDeltaTransformSpec {
    pub tokens: usize,
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub epsilon: f32,
    pub norm_weight_shift: f32,
}

#[derive(Clone, Debug)]
pub struct GatedDeltaTransforms {
    split_normalize: TypedKernel<SplitNormalizeKernel>,
    norm_gate: TypedKernel<NormGateKernel>,
    spec: GatedDeltaTransformSpec,
}

impl GatedDeltaTransforms {
    pub fn compile(compiler: &Compiler, spec: GatedDeltaTransformSpec) -> Result<Self> {
        validate(spec)?;
        let source = cuda_kernel_file!("../../../kernels/gated_delta_transform_bf16.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self {
            split_normalize: module.kernel()?,
            norm_gate: module.kernel()?,
            spec,
        })
    }

    pub fn split_normalize(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        normalized_query: &mut DeviceBuffer<bf16>,
        normalized_key: &mut DeviceBuffer<bf16>,
        value: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let key_width = product(self.spec.key_heads, self.spec.key_dim)?;
        let value_width = product(self.spec.value_heads, self.spec.value_dim)?;
        let width = key_width
            .checked_mul(2)
            .and_then(|value| value.checked_add(value_width))
            .ok_or(Error::InvalidDecoderKernel("Gated Delta projection width overflow"))?;
        let rows = product(self.spec.tokens, self.spec.key_heads)?;
        let elements = product(rows, self.spec.key_dim)?;
        require("Gated Delta mixed projection", product(self.spec.tokens, width)?, input.len())?;
        require("Gated Delta normalized query", elements, normalized_query.len())?;
        require("Gated Delta normalized key", elements, normalized_key.len())?;
        require("Gated Delta split value", product(self.spec.tokens, value_width)?, value.len())?;
        Ok(self.split_normalize.launch(
            stream,
            LaunchConfig {
                grid: (narrow(rows)?, 1, 1),
                block: (256, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                input,
                normalized_query,
                normalized_key,
                value,
                narrow(self.spec.tokens)?,
                narrow(self.spec.key_heads)?,
                narrow(self.spec.value_heads)?,
                narrow(self.spec.key_dim)?,
                narrow(self.spec.value_dim)?,
                1.0e-6,
            ),
        )?)
    }

    pub fn norm_gate(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        gate: &DeviceBuffer<bf16>,
        weight: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.norm_gate_strided(stream, input, gate, weight, output, self.value_width()?, 0)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn norm_gate_strided(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        gate: &DeviceBuffer<bf16>,
        weight: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        gate_stride: usize,
        gate_offset: usize,
    ) -> Result<()> {
        let rows = product(self.spec.tokens, self.spec.value_heads)?;
        let elements = product(rows, self.spec.value_dim)?;
        require("Gated Delta norm input", elements, input.len())?;
        validate_gate(self.spec, gate, gate_stride, gate_offset)?;
        require("Gated Delta norm weight", self.spec.value_dim, weight.len())?;
        require("Gated Delta gated output", elements, output.len())?;
        Ok(self.norm_gate.launch(
            stream,
            LaunchConfig {
                grid: (narrow(rows)?, 1, 1),
                block: (256, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                input,
                gate,
                weight,
                output,
                narrow(rows)?,
                narrow(self.spec.value_dim)?,
                self.spec.epsilon,
                self.spec.norm_weight_shift,
                narrow(self.spec.value_heads)?,
                narrow(gate_stride)?,
                narrow(gate_offset)?,
            ),
        )?)
    }

    fn value_width(&self) -> Result<usize> {
        product(self.spec.value_heads, self.spec.value_dim)
    }
}

fn validate_gate(
    spec: GatedDeltaTransformSpec,
    gate: &DeviceBuffer<bf16>,
    stride: usize,
    offset: usize,
) -> Result<()> {
    let width = product(spec.value_heads, spec.value_dim)?;
    let row_end = offset
        .checked_add(width)
        .filter(|end| *end <= stride)
        .ok_or(Error::InvalidDecoderKernel("invalid Gated Delta gate stride"))?;
    let required = product(spec.tokens.saturating_sub(1), stride)?
        .checked_add(row_end)
        .ok_or(Error::InvalidDecoderKernel("Gated Delta gate stride overflow"))?;
    if gate.len() < required {
        return Err(Error::InvalidDecoderKernel("strided Gated Delta gate is too small"));
    }
    Ok(())
}

fn validate(spec: GatedDeltaTransformSpec) -> Result<()> {
    if spec.tokens == 0
        || spec.key_heads == 0
        || spec.value_heads == 0
        || !spec.value_heads.is_multiple_of(spec.key_heads)
        || spec.key_dim == 0
        || spec.value_dim == 0
        || !spec.epsilon.is_finite()
        || spec.epsilon < 0.0
        || !spec.norm_weight_shift.is_finite()
    {
        return Err(Error::InvalidDecoderKernel("invalid Gated Delta transform geometry"));
    }
    Ok(())
}
