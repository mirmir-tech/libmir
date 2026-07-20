use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::super::geometry::{narrow, product, require};
use crate::{Error, Result};

cuda_export!(
    ConvertKernel = "libmir_cuda_vision_convert_f32_bf16"(
        input: &DeviceBuffer<f32>, output: &mut DeviceBuffer<bf16>, elements: u32,
        scale: f32, bias: f32,
    )
);
cuda_export!(
    BinaryKernel = "libmir_cuda_vision_binary_bf16"(
        left: &DeviceBuffer<bf16>, right: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>, elements: u32, operation: u32,
    )
);
cuda_export!(
    BiasKernel = "libmir_cuda_vision_bias_bf16"(
        input: &DeviceBuffer<bf16>, bias: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>, rows: u32, columns: u32,
    )
);
cuda_export!(
    GeluKernel = "libmir_cuda_vision_gelu_bf16"(
        input: &DeviceBuffer<bf16>, output: &mut DeviceBuffer<bf16>,
        elements: u32, approximate: u32,
    )
);
cuda_export!(
    LayerNormKernel = "libmir_cuda_vision_layer_norm_bf16"(
        input: &DeviceBuffer<bf16>, weight: &DeviceBuffer<bf16>,
        bias: &DeviceBuffer<bf16>, output: &mut DeviceBuffer<bf16>,
        rows: u32, columns: u32, epsilon: f32,
    )
);
cuda_export!(
    PositionKernel = "libmir_cuda_vision_position_add_bf16"(
        input: &DeviceBuffer<bf16>, table: &DeviceBuffer<bf16>,
        positions: &DeviceBuffer<u32>, output: &mut DeviceBuffer<bf16>,
        tokens: u32, positions_per_axis: u32, hidden: u32,
    )
);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VisionElementwiseSpec {
    pub rows: usize,
    pub columns: usize,
    pub epsilon: f32,
}

#[derive(Clone, Debug)]
pub struct VisionElementwise {
    convert: TypedKernel<ConvertKernel>,
    binary: TypedKernel<BinaryKernel>,
    bias: TypedKernel<BiasKernel>,
    gelu: TypedKernel<GeluKernel>,
    layer_norm: TypedKernel<LayerNormKernel>,
    position: TypedKernel<PositionKernel>,
    spec: VisionElementwiseSpec,
}

impl VisionElementwise {
    pub fn compile(compiler: &Compiler, spec: VisionElementwiseSpec) -> Result<Self> {
        if spec.rows == 0 || spec.columns == 0 || !spec.epsilon.is_finite() || spec.epsilon < 0.0 {
            return Err(Error::InvalidVisionKernel("invalid elementwise geometry"));
        }
        let source = cuda_kernel_file!("../../../kernels/vision_bf16.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self {
            convert: module.kernel()?,
            binary: module.kernel()?,
            bias: module.kernel()?,
            gelu: module.kernel()?,
            layer_norm: module.kernel()?,
            position: module.kernel()?,
            spec,
        })
    }

    pub fn convert(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<bf16>,
        scale: f32,
        bias: f32,
    ) -> Result<()> {
        let elements = self.elements()?;
        require("vision convert input", elements, input.len())?;
        require("vision convert output", elements, output.len())?;
        Ok(self.convert.launch(
            stream,
            launch(elements)?,
            (input, output, narrow(elements)?, scale, bias),
        )?)
    }

    pub fn add(
        &self,
        stream: &Stream,
        left: &DeviceBuffer<bf16>,
        right: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.binary(stream, left, right, output, 0)
    }

    pub fn multiply(
        &self,
        stream: &Stream,
        left: &DeviceBuffer<bf16>,
        right: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.binary(stream, left, right, output, 1)
    }

    pub fn add_bias(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        bias: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.validate(input, output)?;
        require("vision bias", self.spec.columns, bias.len())?;
        Ok(self.bias.launch(
            stream,
            launch(self.elements()?)?,
            (input, bias, output, narrow(self.spec.rows)?, narrow(self.spec.columns)?),
        )?)
    }

    pub fn gelu(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        approximate: bool,
    ) -> Result<()> {
        self.validate(input, output)?;
        Ok(self.gelu.launch(
            stream,
            launch(self.elements()?)?,
            (input, output, narrow(self.elements()?)?, u32::from(approximate)),
        )?)
    }

    pub fn layer_norm(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        weight: &DeviceBuffer<bf16>,
        bias: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.validate(input, output)?;
        require("vision norm weight", self.spec.columns, weight.len())?;
        require("vision norm bias", self.spec.columns, bias.len())?;
        Ok(self.layer_norm.launch(
            stream,
            LaunchConfig {
                grid: (narrow(self.spec.rows)?, 1, 1),
                block: (256, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                input,
                weight,
                bias,
                output,
                narrow(self.spec.rows)?,
                narrow(self.spec.columns)?,
                self.spec.epsilon,
            ),
        )?)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_positions(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        table: &DeviceBuffer<bf16>,
        positions: &DeviceBuffer<u32>,
        positions_per_axis: usize,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.validate(input, output)?;
        require("vision positions", self.spec.rows * 2, positions.len())?;
        require("vision position table", 2 * positions_per_axis * self.spec.columns, table.len())?;
        Ok(self.position.launch(
            stream,
            launch(self.elements()?)?,
            (
                input,
                table,
                positions,
                output,
                narrow(self.spec.rows)?,
                narrow(positions_per_axis)?,
                narrow(self.spec.columns)?,
            ),
        )?)
    }

    fn binary(
        &self,
        stream: &Stream,
        left: &DeviceBuffer<bf16>,
        right: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        operation: u32,
    ) -> Result<()> {
        self.validate(left, output)?;
        require("vision binary right", self.elements()?, right.len())?;
        Ok(self.binary.launch(
            stream,
            launch(self.elements()?)?,
            (left, right, output, narrow(self.elements()?)?, operation),
        )?)
    }

    fn validate(&self, input: &DeviceBuffer<bf16>, output: &DeviceBuffer<bf16>) -> Result<()> {
        require("vision input", self.elements()?, input.len())?;
        require("vision output", self.elements()?, output.len())
    }

    fn elements(&self) -> Result<usize> {
        product(self.spec.rows, self.spec.columns)
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
