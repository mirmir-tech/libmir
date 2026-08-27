use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::{GatedActivation, geometry::require};
use crate::{Error, Result};

mod reductions;

cuda_export!(
    AddKernel = "libmir_cuda_add_bf16"(
        left: &DeviceBuffer<bf16>, right: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>, elements: u32,
    )
);
cuda_export!(
    WeightedReduceBucketedResidualSharedKernel =
        "libmir_cuda_weighted_reduce_bucketed_residual_shared_bf16"(
            input: &DeviceBuffer<bf16>, weights: &DeviceBuffer<bf16>,
            positions: &DeviceBuffer<u32>, residual: &DeviceBuffer<bf16>,
            shared: &DeviceBuffer<bf16>, output: &mut DeviceBuffer<bf16>,
            rows: u32, columns: u32, tokens: u32,
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
cuda_export!(
    PackedGatedKernel = "libmir_cuda_packed_gated_bf16"(
        gate: &DeviceBuffer<bf16>, up: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>, columns: u32, elements: u32,
        layout: u32, activation: u32,
    )
);
#[derive(Clone, Debug)]
pub struct ElementwiseBf16 {
    add: TypedKernel<AddKernel>,
    multiply_scalar: TypedKernel<MultiplyScalarKernel>,
    gated: TypedKernel<GatedKernel>,
    packed_gated: TypedKernel<PackedGatedKernel>,
    weighted_reduce: TypedKernel<WeightedReduceKernel>,
    weighted_reduce_bucketed: TypedKernel<WeightedReduceBucketedKernel>,
    weighted_reduce_bucketed_residual_shared:
        TypedKernel<WeightedReduceBucketedResidualSharedKernel>,
    elements: usize,
}

impl ElementwiseBf16 {
    pub fn compile(compiler: &Compiler, elements: usize) -> Result<Self> {
        if elements == 0 {
            return Err(Error::InvalidDecoderKernel("empty elementwise geometry"));
        }
        let source = cuda_kernel_file!("../../../kernels/elementwise_bf16.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self {
            add: module.kernel()?,
            multiply_scalar: module.kernel()?,
            gated: module.kernel()?,
            packed_gated: module.kernel()?,
            weighted_reduce: module.kernel()?,
            weighted_reduce_bucketed: module.kernel()?,
            weighted_reduce_bucketed_residual_shared: module.kernel()?,
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

    pub fn gated_interleaved(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        columns: usize,
        activation: GatedActivation,
    ) -> Result<()> {
        self.gated_packed(stream, input, output, columns, activation, 2)
    }

    pub fn gated_concatenated(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        columns: usize,
        activation: GatedActivation,
    ) -> Result<()> {
        self.gated_packed(stream, input, output, columns, activation, 0)
    }

    fn gated_packed(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        columns: usize,
        activation: GatedActivation,
        layout: u32,
    ) -> Result<()> {
        let packed_elements = self
            .elements
            .checked_mul(2)
            .ok_or(Error::InvalidDecoderKernel("packed gated size overflow"))?;
        require("packed gated input", packed_elements, input.len())?;
        require("packed gated output", self.elements, output.len())?;
        if columns == 0 || !self.elements.is_multiple_of(columns) {
            return Err(Error::InvalidDecoderKernel("invalid packed gated columns"));
        }
        let activation = match activation {
            GatedActivation::GeluTanh => 0,
            GatedActivation::Silu => 1,
        };
        let threads = 256_usize;
        let rows = self.elements / columns;
        Ok(self.packed_gated.launch(
            stream,
            LaunchConfig {
                grid: (u32::try_from(columns.div_ceil(threads))?, u32::try_from(rows)?, 1),
                block: (u32::try_from(threads)?, 1, 1),
                shared_memory_bytes: 0,
            },
            (input, input, output, u32::try_from(columns)?, self.count()?, layout, activation),
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
