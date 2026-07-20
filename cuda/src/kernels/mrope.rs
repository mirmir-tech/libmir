use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::geometry::{narrow, product, require};
use crate::{Error, Result};

cuda_export!(
    MropeKernel = "libmir_cuda_mrope_bf16"(
        input: &DeviceBuffer<bf16>, positions: &DeviceBuffer<u32>,
        output: &mut DeviceBuffer<bf16>, tokens: u32, heads: u32,
        head_dim: u32, rotary_dim: u32, section_t: u32, section_h: u32,
        section_w: u32, interleaved: u32, theta: f32,
    )
);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MropeSpec {
    pub tokens: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub sections: [usize; 3],
    pub interleaved: bool,
    pub theta: f32,
}

#[derive(Clone, Debug)]
pub struct Mrope {
    kernel: TypedKernel<MropeKernel>,
    spec: MropeSpec,
}

impl Mrope {
    pub fn compile(compiler: &Compiler, spec: MropeSpec) -> Result<Self> {
        validate(spec)?;
        let source = cuda_kernel_file!("../../kernels/mrope_bf16.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self { kernel: module.kernel()?, spec })
    }

    pub fn execute(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        positions: &DeviceBuffer<u32>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let elements = product(product(self.spec.tokens, self.spec.heads)?, self.spec.head_dim)?;
        require("MRoPE input", elements, input.len())?;
        require("MRoPE positions", product(self.spec.tokens, 3)?, positions.len())?;
        require("MRoPE output", elements, output.len())?;
        Ok(self.kernel.launch(
            stream,
            LaunchConfig {
                grid: (narrow(elements.div_ceil(256))?, 1, 1),
                block: (256, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                input,
                positions,
                output,
                narrow(self.spec.tokens)?,
                narrow(self.spec.heads)?,
                narrow(self.spec.head_dim)?,
                narrow(self.spec.rotary_dim)?,
                narrow(self.spec.sections[0])?,
                narrow(self.spec.sections[1])?,
                narrow(self.spec.sections[2])?,
                u32::from(self.spec.interleaved),
                self.spec.theta,
            ),
        )?)
    }
}

fn validate(spec: MropeSpec) -> Result<()> {
    let covered = spec.sections.iter().try_fold(0_usize, |total, section| {
        total
            .checked_add(*section)
            .ok_or(Error::InvalidDecoderKernel("MRoPE sections overflow"))
    })?;
    if spec.tokens == 0
        || spec.heads == 0
        || spec.head_dim == 0
        || spec.rotary_dim == 0
        || spec.rotary_dim > spec.head_dim
        || !spec.rotary_dim.is_multiple_of(2)
        || covered != spec.rotary_dim / 2
        || !spec.theta.is_finite()
        || spec.theta <= 0.0
    {
        return Err(Error::InvalidDecoderKernel("invalid MRoPE geometry"));
    }
    Ok(())
}
