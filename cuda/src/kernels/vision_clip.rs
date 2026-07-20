use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::geometry::{narrow, product, require};
use crate::{CudaTensor, Error, Result};

cuda_export!(
    ClipBf16Kernel = "libmir_cuda_vision_clip_bounds_bf16"(
        input: &DeviceBuffer<bf16>, minimum: &DeviceBuffer<bf16>,
        maximum: &DeviceBuffer<bf16>, output: &mut DeviceBuffer<bf16>,
        elements: u32, columns: u32, bounds: u32,
    )
);
cuda_export!(
    ClipF32Kernel = "libmir_cuda_vision_clip_bounds_f32"(
        input: &DeviceBuffer<bf16>, minimum: &DeviceBuffer<f32>,
        maximum: &DeviceBuffer<f32>, output: &mut DeviceBuffer<bf16>,
        elements: u32, columns: u32, bounds: u32,
    )
);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VisionClipSpec {
    pub rows: usize,
    pub columns: usize,
}

#[derive(Clone, Debug)]
pub struct VisionClip {
    bf16: TypedKernel<ClipBf16Kernel>,
    f32: TypedKernel<ClipF32Kernel>,
    spec: VisionClipSpec,
}

impl VisionClip {
    pub fn compile(compiler: &Compiler, spec: VisionClipSpec) -> Result<Self> {
        if spec.rows == 0 || spec.columns == 0 {
            return Err(Error::InvalidVisionKernel("invalid clipping geometry"));
        }
        let module = compiler.compile(
            cuda_kernel_file!("../../kernels/vision_clip_bf16.cu"),
            &CompileOptions::default(),
        )?;
        Ok(Self {
            bf16: module.kernel()?,
            f32: module.kernel()?,
            spec,
        })
    }

    pub fn execute(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        minimum: &CudaTensor,
        maximum: &CudaTensor,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let elements = product(self.spec.rows, self.spec.columns)?;
        require("vision clip input", elements, input.len())?;
        require("vision clip output", elements, output.len())?;
        if minimum.shape() != maximum.shape() {
            return Err(Error::InvalidVisionKernel("clipping bound shapes differ"));
        }
        let bounds = minimum.shape().iter().try_fold(1_usize, |total, value| {
            total
                .checked_mul(*value)
                .ok_or(Error::InvalidVisionKernel("clip size overflow"))
        })?;
        if bounds != 1 && bounds != self.spec.columns {
            return Err(Error::InvalidVisionKernel("clipping bounds must be scalar or per-column"));
        }
        let arguments = (narrow(elements)?, narrow(self.spec.columns)?, narrow(bounds)?);
        let launch = launch(elements)?;
        match (minimum.as_bf16(), maximum.as_bf16()) {
            (Some(minimum), Some(maximum)) => Ok(self.bf16.launch(
                stream,
                launch,
                (input, minimum, maximum, output, arguments.0, arguments.1, arguments.2),
            )?),
            _ => match (minimum.as_f32(), maximum.as_f32()) {
                (Some(minimum), Some(maximum)) => Ok(self.f32.launch(
                    stream,
                    launch,
                    (input, minimum, maximum, output, arguments.0, arguments.1, arguments.2),
                )?),
                _ => Err(Error::DTypeMismatch {
                    name: minimum.name().into(),
                    expected: "matching BF16 or F32 clipping bounds",
                }),
            },
        }
    }
}

fn launch(elements: usize) -> Result<LaunchConfig> {
    Ok(LaunchConfig {
        grid: (narrow(elements.div_ceil(256))?, 1, 1),
        block: (256, 1, 1),
        shared_memory_bytes: 0,
    })
}
