use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::geometry::{narrow, product, require};
use crate::{Error, Result};

cuda_export!(PerLayerInputKernel = "libmir_cuda_per_layer_input_bf16"(
    projected: &DeviceBuffer<bf16>, embedded: &DeviceBuffer<bf16>, weight: &DeviceBuffer<bf16>,
    output: &mut DeviceBuffer<bf16>, rows: u32, layers: u32, width: u32, projection_scale: f32,
    input_scale: f32, epsilon: f32,
));

/// The shape and scales of Gemma 4's per-layer input combination.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PerLayerInputSpec {
    pub rows: usize,
    pub layers: usize,
    pub width: usize,
    /// On the projection of the token embeddings, before its norm.
    pub projection_scale: f32,
    /// On the sum of the normalized projection and the per-layer embedding.
    pub input_scale: f32,
    pub epsilon: f32,
}

/// Combines the normalized projection of the token embeddings with the
/// per-layer embeddings, both `[rows, layers * width]`, into layer-major
/// `[layers, rows, width]` inputs.
#[derive(Clone, Debug)]
pub struct PerLayerInputBf16 {
    kernel: TypedKernel<PerLayerInputKernel>,
    spec: PerLayerInputSpec,
}

impl PerLayerInputBf16 {
    pub fn compile(compiler: &Compiler, spec: PerLayerInputSpec) -> Result<Self> {
        if spec.rows == 0 || spec.layers == 0 || spec.width == 0 {
            return Err(Error::InvalidDecoderKernel("empty per-layer input geometry"));
        }
        let module = compiler.compile(
            cuda_kernel_file!("../../kernels/per_layer_input_bf16.cu"),
            &CompileOptions::default(),
        )?;
        Ok(Self { kernel: module.kernel()?, spec })
    }

    pub fn execute(
        &self,
        stream: &Stream,
        projected: &DeviceBuffer<bf16>,
        embedded: &DeviceBuffer<bf16>,
        weight: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let spec = self.spec;
        let elements = product(product(spec.rows, spec.layers)?, spec.width)?;
        require("per-layer projection", elements, projected.len())?;
        require("per-layer embedding", elements, embedded.len())?;
        require("per-layer projection norm", spec.width, weight.len())?;
        require("per-layer inputs", elements, output.len())?;
        let threads = spec.width.next_power_of_two().clamp(32, 1_024);
        Ok(self.kernel.launch(
            stream,
            LaunchConfig {
                grid: (narrow(spec.layers)?, narrow(spec.rows)?, 1),
                block: (narrow(threads)?, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                projected,
                embedded,
                weight,
                output,
                narrow(spec.rows)?,
                narrow(spec.layers)?,
                narrow(spec.width)?,
                spec.projection_scale,
                spec.input_scale,
                spec.epsilon,
            ),
        )?)
    }
}
