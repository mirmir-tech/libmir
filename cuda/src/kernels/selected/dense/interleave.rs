use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use crate::{
    Error, Result,
    kernels::geometry::{narrow, product, require},
};

cuda_export!(InterleaveKernel = "libmir_cuda_dense_expert_interleave_bf16"(
    gate: &DeviceBuffer<bf16>, up: &DeviceBuffer<bf16>, output: &mut DeviceBuffer<bf16>,
    rows: u32, columns: u32,
));

/// Gate and up banks to one fused bank with gate row `r` at `2r` and up row
/// `r` at `2r + 1`: the layout of the dense expert GEMM's tensor-core path.
#[derive(Clone, Debug)]
pub struct DenseExpertInterleave {
    kernel: TypedKernel<InterleaveKernel>,
}

impl DenseExpertInterleave {
    pub fn compile(compiler: &Compiler) -> Result<Self> {
        let source = cuda_kernel_file!("../../../../kernels/dense_expert_interleave_bf16.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self { kernel: module.kernel()? })
    }

    pub fn execute(
        &self,
        stream: &Stream,
        gate: &DeviceBuffer<bf16>,
        up: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        rows: usize,
        columns: usize,
    ) -> Result<()> {
        if !columns.is_multiple_of(8) {
            return Err(Error::InvalidDecoderKernel(
                "interleaved expert rows need 8-value columns",
            ));
        }
        let elements = product(rows, columns)?;
        require("expert gate bank", elements, gate.len())?;
        require("expert up bank", elements, up.len())?;
        require("fused expert bank", product(elements, 2)?, output.len())?;
        let threads = product(elements / 8, 2)?;
        self.kernel.launch(
            stream,
            LaunchConfig {
                grid: (narrow(threads.div_ceil(256))?, 1, 1),
                block: (256, 1, 1),
                shared_memory_bytes: 0,
            },
            (gate, up, output, narrow(rows)?, narrow(columns)?),
        )?;
        Ok(())
    }
}
