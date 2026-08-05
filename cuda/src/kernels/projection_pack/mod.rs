use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use crate::{Error, Result, kernels::geometry::narrow};

cuda_export!(Split2Kernel = "libmir_cuda_projection_pack_split2_bf16"(
    input: &DeviceBuffer<bf16>, first: &mut DeviceBuffer<bf16>,
    second: &mut DeviceBuffer<bf16>, first_columns: u32,
    second_columns: u32, tokens: u32,
));
cuda_export!(Split3Kernel = "libmir_cuda_projection_pack_split3_bf16"(
    input: &DeviceBuffer<bf16>, first: &mut DeviceBuffer<bf16>,
    second: &mut DeviceBuffer<bf16>, third: &mut DeviceBuffer<bf16>,
    first_columns: u32, second_columns: u32, third_columns: u32, tokens: u32,
));

#[derive(Clone, Debug)]
pub struct ProjectionPackSplit {
    split2: TypedKernel<Split2Kernel>,
    split3: TypedKernel<Split3Kernel>,
    tokens: usize,
    columns: Vec<usize>,
}

impl ProjectionPackSplit {
    pub fn compile(compiler: &Compiler, tokens: usize, columns: &[usize]) -> Result<Self> {
        if tokens == 0 || !matches!(columns.len(), 2 | 3) || columns.contains(&0) {
            return Err(Error::InvalidDecoderKernel("invalid projection pack split geometry"));
        }
        let module = compiler.compile(
            cuda_kernel_file!("../../../kernels/projection_pack_bf16.cu"),
            &CompileOptions::default(),
        )?;
        Ok(Self {
            split2: module.kernel()?,
            split3: module.kernel()?,
            tokens,
            columns: columns.to_vec(),
        })
    }

    pub fn execute2(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        first: &mut DeviceBuffer<bf16>,
        second: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.validate(input, &[first.len(), second.len()])?;
        Ok(self.split2.launch(
            stream,
            Self::launch(input.len())?,
            (
                input,
                first,
                second,
                narrow(self.columns[0])?,
                narrow(self.columns[1])?,
                narrow(self.tokens)?,
            ),
        )?)
    }

    pub fn execute3(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        first: &mut DeviceBuffer<bf16>,
        second: &mut DeviceBuffer<bf16>,
        third: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.validate(input, &[first.len(), second.len(), third.len()])?;
        Ok(self.split3.launch(
            stream,
            Self::launch(input.len())?,
            (
                input,
                first,
                second,
                third,
                narrow(self.columns[0])?,
                narrow(self.columns[1])?,
                narrow(self.columns[2])?,
                narrow(self.tokens)?,
            ),
        )?)
    }

    fn validate(&self, input: &DeviceBuffer<bf16>, outputs: &[usize]) -> Result<()> {
        let columns = self.columns.iter().sum::<usize>();
        let valid = input.len() == self.tokens * columns
            && outputs.len() == self.columns.len()
            && outputs
                .iter()
                .zip(&self.columns)
                .all(|(actual, columns)| *actual == self.tokens * columns);
        if valid {
            Ok(())
        } else {
            Err(Error::InvalidDecoderKernel("projection pack split buffer mismatch"))
        }
    }

    fn launch(elements: usize) -> Result<LaunchConfig> {
        let threads = 256_usize;
        Ok(LaunchConfig {
            grid: (narrow(elements.div_ceil(threads))?, 1, 1),
            block: (narrow(threads)?, 1, 1),
            shared_memory_bytes: 0,
        })
    }
}
