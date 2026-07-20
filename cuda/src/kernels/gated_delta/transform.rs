use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::super::geometry::{narrow, product, require};
use crate::{Error, Result};

cuda_export!(
    SplitKernel = "libmir_cuda_gated_delta_split_bf16"(
        input: &DeviceBuffer<bf16>, query: &mut DeviceBuffer<bf16>,
        key: &mut DeviceBuffer<bf16>, value: &mut DeviceBuffer<bf16>,
        tokens: u32, key_width: u32, value_width: u32,
    )
);

cuda_export!(
    NormalizeQkKernel = "libmir_cuda_gated_delta_normalize_qk_bf16"(
        query: &DeviceBuffer<bf16>, key: &DeviceBuffer<bf16>,
        normalized_query: &mut DeviceBuffer<bf16>,
        normalized_key: &mut DeviceBuffer<bf16>, rows: u32, columns: u32,
        epsilon: f32,
    )
);

cuda_export!(
    NormGateKernel = "libmir_cuda_gated_delta_norm_gate_bf16"(
        input: &DeviceBuffer<bf16>, gate: &DeviceBuffer<bf16>,
        weight: &DeviceBuffer<bf16>, output: &mut DeviceBuffer<bf16>,
        rows: u32, columns: u32, epsilon: f32, weight_shift: f32,
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
    split: TypedKernel<SplitKernel>,
    normalize_qk: TypedKernel<NormalizeQkKernel>,
    norm_gate: TypedKernel<NormGateKernel>,
    spec: GatedDeltaTransformSpec,
}

impl GatedDeltaTransforms {
    pub fn compile(compiler: &Compiler, spec: GatedDeltaTransformSpec) -> Result<Self> {
        validate(spec)?;
        let source = cuda_kernel_file!("../../../kernels/gated_delta_transform_bf16.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self {
            split: module.kernel()?,
            normalize_qk: module.kernel()?,
            norm_gate: module.kernel()?,
            spec,
        })
    }

    pub fn split(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        query: &mut DeviceBuffer<bf16>,
        key: &mut DeviceBuffer<bf16>,
        value: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let key_width = product(self.spec.key_heads, self.spec.key_dim)?;
        let value_width = product(self.spec.value_heads, self.spec.value_dim)?;
        let width = key_width
            .checked_mul(2)
            .and_then(|value| value.checked_add(value_width))
            .ok_or(Error::InvalidDecoderKernel("Gated Delta projection width overflow"))?;
        let elements = product(self.spec.tokens, width)?;
        require("Gated Delta mixed projection", elements, input.len())?;
        require("Gated Delta split query", product(self.spec.tokens, key_width)?, query.len())?;
        require("Gated Delta split key", product(self.spec.tokens, key_width)?, key.len())?;
        require("Gated Delta split value", product(self.spec.tokens, value_width)?, value.len())?;
        let threads = 256_usize;
        Ok(self.split.launch(
            stream,
            LaunchConfig {
                grid: (narrow(elements.div_ceil(threads))?, 1, 1),
                block: (narrow(threads)?, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                input,
                query,
                key,
                value,
                narrow(self.spec.tokens)?,
                narrow(key_width)?,
                narrow(value_width)?,
            ),
        )?)
    }

    pub fn normalize_qk(
        &self,
        stream: &Stream,
        query: &DeviceBuffer<bf16>,
        key: &DeviceBuffer<bf16>,
        normalized_query: &mut DeviceBuffer<bf16>,
        normalized_key: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let rows = product(self.spec.tokens, self.spec.key_heads)?;
        let elements = product(rows, self.spec.key_dim)?;
        require("Gated Delta query normalization input", elements, query.len())?;
        require("Gated Delta key normalization input", elements, key.len())?;
        require("Gated Delta normalized query", elements, normalized_query.len())?;
        require("Gated Delta normalized key", elements, normalized_key.len())?;
        Ok(self.normalize_qk.launch(
            stream,
            LaunchConfig {
                grid: (narrow(rows)?, 1, 1),
                block: (256, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                query,
                key,
                normalized_query,
                normalized_key,
                narrow(rows)?,
                narrow(self.spec.key_dim)?,
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
        let rows = product(self.spec.tokens, self.spec.value_heads)?;
        let elements = product(rows, self.spec.value_dim)?;
        require("Gated Delta norm input", elements, input.len())?;
        require("Gated Delta gate", elements, gate.len())?;
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
            ),
        )?)
    }
}

fn validate(spec: GatedDeltaTransformSpec) -> Result<()> {
    if spec.tokens == 0
        || spec.key_heads == 0
        || spec.value_heads == 0
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
