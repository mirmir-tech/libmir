use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::{GatedActivation, geometry::require};
use crate::{Error, Result};

cuda_export!(
    AddKernel = "libmir_cuda_add_bf16"(
        left: &DeviceBuffer<bf16>, right: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>, elements: u32,
    )
);
cuda_export!(
    WeightedReduceKernel = "libmir_cuda_weighted_reduce_bf16"(
        input: &DeviceBuffer<bf16>, weights: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>, rows: u32, columns: u32, tokens: u32,
    )
);
cuda_export!(
    WeightedReduceBucketedKernel = "libmir_cuda_weighted_reduce_bucketed_bf16"(
        input: &DeviceBuffer<bf16>, weights: &DeviceBuffer<bf16>,
        positions: &DeviceBuffer<u32>, output: &mut DeviceBuffer<bf16>,
        rows: u32, columns: u32, tokens: u32,
    )
);
cuda_export!(
    MultiplyScalarKernel = "libmir_cuda_multiply_scalar_bf16"(
        input: &DeviceBuffer<bf16>, scalar: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>, elements: u32,
    )
);
cuda_export!(
    GatedKernel = "libmir_cuda_gated_bf16"(
        gate: &DeviceBuffer<bf16>, up: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>, elements: u32, activation: u32,
    )
);

#[derive(Clone, Debug)]
pub struct ElementwiseBf16 {
    add: TypedKernel<AddKernel>,
    multiply_scalar: TypedKernel<MultiplyScalarKernel>,
    gated: TypedKernel<GatedKernel>,
    weighted_reduce: TypedKernel<WeightedReduceKernel>,
    weighted_reduce_bucketed: TypedKernel<WeightedReduceBucketedKernel>,
    elements: usize,
}

impl ElementwiseBf16 {
    pub fn compile(compiler: &Compiler, elements: usize) -> Result<Self> {
        if elements == 0 {
            return Err(Error::InvalidDecoderKernel("empty elementwise geometry"));
        }
        let source = cuda_kernel_file!("../../kernels/elementwise_bf16.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self {
            add: module.kernel()?,
            multiply_scalar: module.kernel()?,
            gated: module.kernel()?,
            weighted_reduce: module.kernel()?,
            weighted_reduce_bucketed: module.kernel()?,
            elements,
        })
    }

    pub fn add(
        &self,
        stream: &Stream,
        left: &DeviceBuffer<bf16>,
        right: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.validate(left, right, output)?;
        Ok(self.add.launch(stream, self.launch()?, (left, right, output, self.count()?))?)
    }

    pub fn multiply_scalar(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        scalar: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        require("elementwise input", self.elements, input.len())?;
        require("elementwise scalar", 1, scalar.len())?;
        require("elementwise output", self.elements, output.len())?;
        Ok(self.multiply_scalar.launch(
            stream,
            self.launch()?,
            (input, scalar, output, self.count()?),
        )?)
    }

    pub fn gated(
        &self,
        stream: &Stream,
        gate: &DeviceBuffer<bf16>,
        up: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        activation: GatedActivation,
    ) -> Result<()> {
        self.validate(gate, up, output)?;
        let activation = match activation {
            GatedActivation::GeluTanh => 0,
            GatedActivation::Silu => 1,
        };
        Ok(self.gated.launch(
            stream,
            self.launch()?,
            (gate, up, output, self.count()?, activation),
        )?)
    }

    pub fn weighted_reduce(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        weights: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        rows: usize,
    ) -> Result<()> {
        self.weighted_reduce_batch(stream, input, weights, output, rows, 1)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn weighted_reduce_batch(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        weights: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        rows: usize,
        tokens: usize,
    ) -> Result<()> {
        let input_elements = self
            .elements
            .checked_mul(rows)
            .and_then(|elements| elements.checked_mul(tokens))
            .ok_or(Error::InvalidDecoderKernel("weighted reduction overflow"))?;
        let weight_elements = rows
            .checked_mul(tokens)
            .ok_or(Error::InvalidDecoderKernel("weighted reduction overflow"))?;
        let output_elements = self
            .elements
            .checked_mul(tokens)
            .ok_or(Error::InvalidDecoderKernel("weighted reduction overflow"))?;
        require("weighted reduction input", input_elements, input.len())?;
        require("weighted reduction weights", weight_elements, weights.len())?;
        require("weighted reduction output", output_elements, output.len())?;
        if tokens == 0 {
            return Err(Error::InvalidDecoderKernel("weighted reduction batch is empty"));
        }
        let threads = 256_usize;
        Ok(self.weighted_reduce.launch(
            stream,
            LaunchConfig {
                grid: (u32::try_from(output_elements.div_ceil(threads))?, 1, 1),
                block: (u32::try_from(threads)?, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                input,
                weights,
                output,
                u32::try_from(rows)?,
                self.count()?,
                u32::try_from(tokens)?,
            ),
        )?)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn weighted_reduce_bucketed(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        weights: &DeviceBuffer<bf16>,
        positions: &DeviceBuffer<u32>,
        output: &mut DeviceBuffer<bf16>,
        rows: usize,
        tokens: usize,
    ) -> Result<()> {
        let assignments = rows
            .checked_mul(tokens)
            .ok_or(Error::InvalidDecoderKernel("bucketed reduction overflow"))?;
        let input_elements = self
            .elements
            .checked_mul(assignments)
            .ok_or(Error::InvalidDecoderKernel("bucketed reduction overflow"))?;
        let output_elements = self
            .elements
            .checked_mul(tokens)
            .ok_or(Error::InvalidDecoderKernel("bucketed reduction overflow"))?;
        require("bucketed reduction input", input_elements, input.len())?;
        require("bucketed reduction weights", assignments, weights.len())?;
        require("bucketed reduction positions", assignments, positions.len())?;
        require("bucketed reduction output", output_elements, output.len())?;
        let threads = 256_usize;
        Ok(self.weighted_reduce_bucketed.launch(
            stream,
            LaunchConfig {
                grid: (u32::try_from(output_elements.div_ceil(threads))?, 1, 1),
                block: (u32::try_from(threads)?, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                input,
                weights,
                positions,
                output,
                u32::try_from(rows)?,
                self.count()?,
                u32::try_from(tokens)?,
            ),
        )?)
    }

    fn validate(
        &self,
        left: &DeviceBuffer<bf16>,
        right: &DeviceBuffer<bf16>,
        output: &DeviceBuffer<bf16>,
    ) -> Result<()> {
        require("elementwise left", self.elements, left.len())?;
        require("elementwise right", self.elements, right.len())?;
        require("elementwise output", self.elements, output.len())
    }

    fn launch(&self) -> Result<LaunchConfig> {
        let threads = 256_usize;
        Ok(LaunchConfig {
            grid: (u32::try_from(self.elements.div_ceil(threads))?, 1, 1),
            block: (u32::try_from(threads)?, 1, 1),
            shared_memory_bytes: 0,
        })
    }

    fn count(&self) -> Result<u32> {
        Ok(u32::try_from(self.elements)?)
    }
}
