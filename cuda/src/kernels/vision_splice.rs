use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::geometry::{narrow, product, require};
use crate::{Error, Result};

cuda_export!(
    SpliceKernel = "libmir_cuda_vision_embedding_splice_bf16"(
        image: &DeviceBuffer<bf16>, hidden: &mut DeviceBuffer<bf16>,
        image_tokens: u32, hidden_width: u32, destination_start: u32,
    )
);

#[derive(Clone, Debug)]
pub struct VisionEmbeddingSplice {
    kernel: TypedKernel<SpliceKernel>,
    image_tokens: usize,
    hidden_width: usize,
    destination_start: usize,
}

impl VisionEmbeddingSplice {
    pub(crate) fn compile(
        compiler: &Compiler,
        image_tokens: usize,
        hidden_width: usize,
        destination_start: usize,
    ) -> Result<Self> {
        if image_tokens == 0 || hidden_width == 0 {
            return Err(Error::InvalidVisionKernel("invalid embedding splice geometry"));
        }
        let module = compiler.compile(
            cuda_kernel_file!("../../kernels/vision_splice_bf16.cu"),
            &CompileOptions::default(),
        )?;
        Ok(Self {
            kernel: module.kernel()?,
            image_tokens,
            hidden_width,
            destination_start,
        })
    }

    pub(crate) fn execute(
        &self,
        stream: &Stream,
        image: &DeviceBuffer<bf16>,
        hidden: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let elements = product(self.image_tokens, self.hidden_width)?;
        let destination = product(self.destination_start + self.image_tokens, self.hidden_width)?;
        require("vision splice image", elements, image.len())?;
        require("vision splice destination", destination, hidden.len())?;
        Ok(self.kernel.launch(
            stream,
            LaunchConfig {
                grid: (narrow(elements.div_ceil(256))?, 1, 1),
                block: (256, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                image,
                hidden,
                narrow(self.image_tokens)?,
                narrow(self.hidden_width)?,
                narrow(self.destination_start)?,
            ),
        )?)
    }
}
