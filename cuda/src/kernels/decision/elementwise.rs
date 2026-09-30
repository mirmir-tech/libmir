use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, Stream, TypedKernel, cuda_export, cuda_kernel_file, f16,
};

use super::{elements_launch, require, rows_launch};
use crate::Result;

cuda_export!(EmbedKernel = "libmir_decision_embed"(
    ids: &DeviceBuffer<u32>, table: &DeviceBuffer<f16>, output: &mut DeviceBuffer<f32>,
    rows: u32, width: u32,
));
cuda_export!(BiasKernel = "libmir_decision_add_bias"(
    values: &mut DeviceBuffer<f32>, bias: &DeviceBuffer<f32>, elements: u32, width: u32,
));
cuda_export!(GeluKernel = "libmir_decision_gelu"(values: &mut DeviceBuffer<f32>, elements: u32));
cuda_export!(ReluKernel = "libmir_decision_relu"(values: &mut DeviceBuffer<f32>, elements: u32));
cuda_export!(KindKernel = "libmir_decision_add_kind"(
    hidden: &mut DeviceBuffer<f32>, kinds: &DeviceBuffer<u32>, table: &DeviceBuffer<f32>,
    elements: u32, length: u32, width: u32,
));
cuda_export!(GatherKernel = "libmir_decision_gather"(
    input: &DeviceBuffer<f32>, rows: &DeviceBuffer<u32>, output: &mut DeviceBuffer<f32>,
    count: u32, width: u32,
));

/// Row-wise and element-wise kernels; every buffer is row-major.
#[derive(Clone, Debug)]
pub struct DecisionElementwise {
    embed: TypedKernel<EmbedKernel>,
    bias: TypedKernel<BiasKernel>,
    gelu: TypedKernel<GeluKernel>,
    relu: TypedKernel<ReluKernel>,
    kind: TypedKernel<KindKernel>,
    gather: TypedKernel<GatherKernel>,
}

impl DecisionElementwise {
    pub fn compile(compiler: &Compiler) -> Result<Self> {
        let source = cuda_kernel_file!("../../../kernels/decision/decision_f32.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self {
            embed: module.kernel()?,
            bias: module.kernel()?,
            gelu: module.kernel()?,
            relu: module.kernel()?,
            kind: module.kernel()?,
            gather: module.kernel()?,
        })
    }

    pub fn embed(
        &self,
        stream: &Stream,
        ids: &DeviceBuffer<u32>,
        table: &DeviceBuffer<f16>,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let width = output.len() / ids.len().max(1);
        require(output, ids.len() * width, "decision embedding output")?;
        let (rows, width) = (u32::try_from(ids.len())?, u32::try_from(width)?);
        Ok(self
            .embed
            .launch(stream, rows_launch(ids.len())?, (ids, table, output, rows, width))?)
    }

    pub fn add_bias(
        &self,
        stream: &Stream,
        values: &mut DeviceBuffer<f32>,
        bias: &DeviceBuffer<f32>,
    ) -> Result<()> {
        let (elements, width) = (u32::try_from(values.len())?, u32::try_from(bias.len())?);
        Ok(self.bias.launch(
            stream,
            elements_launch(values.len())?,
            (values, bias, elements, width),
        )?)
    }

    pub fn gelu(&self, stream: &Stream, values: &mut DeviceBuffer<f32>) -> Result<()> {
        let elements = u32::try_from(values.len())?;
        Ok(self.gelu.launch(stream, elements_launch(values.len())?, (values, elements))?)
    }

    pub fn relu(&self, stream: &Stream, values: &mut DeviceBuffer<f32>) -> Result<()> {
        let elements = u32::try_from(values.len())?;
        Ok(self.relu.launch(stream, elements_launch(values.len())?, (values, elements))?)
    }

    /// Adds the kind embedding of each sequence to all its `length` tokens.
    pub fn add_kind(
        &self,
        stream: &Stream,
        hidden: &mut DeviceBuffer<f32>,
        (kinds, table, length): (&DeviceBuffer<u32>, &DeviceBuffer<f32>, usize),
    ) -> Result<()> {
        let width = table.len() / 3;
        let geometry =
            (u32::try_from(hidden.len())?, u32::try_from(length)?, u32::try_from(width)?);
        let launch = elements_launch(hidden.len())?;
        Ok(self.kind.launch(
            stream,
            launch,
            (hidden, kinds, table, geometry.0, geometry.1, geometry.2),
        )?)
    }

    pub fn gather(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        rows: &DeviceBuffer<u32>,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let width = output.len() / rows.len().max(1);
        require(output, rows.len() * width, "decision gather output")?;
        let geometry = (u32::try_from(rows.len())?, u32::try_from(width)?);
        Ok(self.gather.launch(
            stream,
            elements_launch(output.len())?,
            (input, rows, output, geometry.0, geometry.1),
        )?)
    }
}
