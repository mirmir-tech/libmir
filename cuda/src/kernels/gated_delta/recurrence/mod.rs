use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::super::geometry::product;
use crate::{Error, Result};

mod chunked;
mod execute;

pub use chunked::{GatedDeltaChunked, GatedDeltaChunkedScratch};

cuda_export!(
    ParametersKernel = "libmir_cuda_gated_delta_parameters_bf16"(
        alpha: &DeviceBuffer<bf16>, beta: &DeviceBuffer<bf16>,
        a_log: &DeviceBuffer<bf16>, dt_bias: &DeviceBuffer<bf16>,
        decay: &mut DeviceBuffer<f32>, update: &mut DeviceBuffer<f32>,
        tokens: u32, value_heads: u32,
    )
);

cuda_export!(
    SerialKernel = "libmir_cuda_gated_delta_recurrence_bf16"(
        query: &DeviceBuffer<bf16>, key: &DeviceBuffer<bf16>, value: &DeviceBuffer<bf16>,
        alpha: &DeviceBuffer<bf16>, beta: &DeviceBuffer<bf16>, a_log: &DeviceBuffer<bf16>,
        dt_bias: &DeviceBuffer<bf16>, decay: &DeviceBuffer<f32>, update: &DeviceBuffer<f32>,
        state: &mut DeviceBuffer<f32>, output: &mut DeviceBuffer<bf16>, tokens: u32,
        key_heads: u32, value_heads: u32, key_dim: u32, value_dim: u32,
    )
);

cuda_export!(
    ValueTiled2Kernel = "libmir_cuda_gated_delta_recurrence_value_tiled_2_bf16"(
        query: &DeviceBuffer<bf16>, key: &DeviceBuffer<bf16>, value: &DeviceBuffer<bf16>,
        decay: &DeviceBuffer<f32>, update: &DeviceBuffer<f32>, state: &mut DeviceBuffer<f32>,
        output: &mut DeviceBuffer<bf16>, tokens: u32, key_heads: u32, value_heads: u32,
        key_dim: u32, value_dim: u32,
    )
);

cuda_export!(
    ValueTiled4Kernel = "libmir_cuda_gated_delta_recurrence_value_tiled_4_bf16"(
        query: &DeviceBuffer<bf16>, key: &DeviceBuffer<bf16>, value: &DeviceBuffer<bf16>,
        decay: &DeviceBuffer<f32>, update: &DeviceBuffer<f32>, state: &mut DeviceBuffer<f32>,
        output: &mut DeviceBuffer<bf16>, tokens: u32, key_heads: u32, value_heads: u32,
        key_dim: u32, value_dim: u32,
    )
);

cuda_export!(
    ValueTiled8Kernel = "libmir_cuda_gated_delta_recurrence_value_tiled_8_bf16"(
        query: &DeviceBuffer<bf16>, key: &DeviceBuffer<bf16>, value: &DeviceBuffer<bf16>,
        decay: &DeviceBuffer<f32>, update: &DeviceBuffer<f32>, state: &mut DeviceBuffer<f32>,
        output: &mut DeviceBuffer<bf16>, tokens: u32, key_heads: u32, value_heads: u32,
        key_dim: u32, value_dim: u32,
    )
);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GatedDeltaRecurrenceMode {
    Serial,
    ValueTiled2,
    ValueTiled4,
    ValueTiled8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GatedDeltaSpec {
    pub tokens: usize,
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
}

pub struct GatedDeltaLaunch<'a> {
    pub query: &'a DeviceBuffer<bf16>,
    pub key: &'a DeviceBuffer<bf16>,
    pub value: &'a DeviceBuffer<bf16>,
    pub alpha: &'a DeviceBuffer<bf16>,
    pub beta: &'a DeviceBuffer<bf16>,
    pub a_log: &'a DeviceBuffer<bf16>,
    pub dt_bias: &'a DeviceBuffer<bf16>,
    pub decay: &'a mut DeviceBuffer<f32>,
    pub update: &'a mut DeviceBuffer<f32>,
    pub state: &'a mut DeviceBuffer<f32>,
    pub output: &'a mut DeviceBuffer<bf16>,
}

#[derive(Clone, Copy)]
pub struct GatedDeltaInputs<'a> {
    pub query: &'a DeviceBuffer<bf16>,
    pub key: &'a DeviceBuffer<bf16>,
    pub value: &'a DeviceBuffer<bf16>,
    pub alpha: &'a DeviceBuffer<bf16>,
    pub beta: &'a DeviceBuffer<bf16>,
    pub a_log: &'a DeviceBuffer<bf16>,
    pub dt_bias: &'a DeviceBuffer<bf16>,
}

#[derive(Clone, Debug)]
pub struct GatedDeltaRecurrence {
    parameters: TypedKernel<ParametersKernel>,
    serial: TypedKernel<SerialKernel>,
    value_tiled_2: TypedKernel<ValueTiled2Kernel>,
    value_tiled_4: TypedKernel<ValueTiled4Kernel>,
    value_tiled_8: TypedKernel<ValueTiled8Kernel>,
    spec: GatedDeltaSpec,
}

impl GatedDeltaRecurrence {
    pub fn compile(compiler: &Compiler, spec: GatedDeltaSpec) -> Result<Self> {
        validate(spec)?;
        let serial = compiler.compile(
            cuda_kernel_file!("../../../../kernels/gated_delta_bf16.cu"),
            &CompileOptions::default(),
        )?;
        let tiled = compiler.compile(
            cuda_kernel_file!("../../../../kernels/gated_delta_recurrence_tiled_bf16.cu"),
            &CompileOptions::default(),
        )?;
        Ok(Self {
            parameters: serial.kernel()?,
            serial: serial.kernel()?,
            value_tiled_2: tiled.kernel()?,
            value_tiled_4: tiled.kernel()?,
            value_tiled_8: tiled.kernel()?,
            spec,
        })
    }

    pub fn execute(&self, stream: &Stream, launch: &mut GatedDeltaLaunch<'_>) -> Result<()> {
        let mode = if self.spec.tokens > 1 && self.spec.key_dim == 128 && self.spec.value_dim == 128
        {
            GatedDeltaRecurrenceMode::ValueTiled2
        } else {
            GatedDeltaRecurrenceMode::Serial
        };
        self.execute_with(stream, launch, mode)
    }

    pub fn execute_with(
        &self,
        stream: &Stream,
        launch: &mut GatedDeltaLaunch<'_>,
        mode: GatedDeltaRecurrenceMode,
    ) -> Result<()> {
        self.validate_launch(launch)?;
        self.prepare_parameters(stream, launch)?;
        match mode {
            GatedDeltaRecurrenceMode::Serial => self.launch_serial(stream, launch),
            GatedDeltaRecurrenceMode::ValueTiled2 => self.launch_value_tiled_2(stream, launch),
            GatedDeltaRecurrenceMode::ValueTiled4 => self.launch_value_tiled_4(stream, launch),
            GatedDeltaRecurrenceMode::ValueTiled8 => self.launch_value_tiled_8(stream, launch),
        }
    }

    pub fn state_elements(&self) -> Result<usize> {
        product(product(self.spec.value_heads, self.spec.value_dim)?, self.spec.key_dim)
    }
}

fn validate(spec: GatedDeltaSpec) -> Result<()> {
    if spec.tokens == 0
        || spec.key_heads == 0
        || spec.value_heads == 0
        || !spec.value_heads.is_multiple_of(spec.key_heads)
        || spec.key_dim == 0
        || !spec.key_dim.is_multiple_of(32)
        || spec.key_dim > 256
        || spec.value_dim == 0
    {
        return Err(Error::InvalidDecoderKernel("invalid Gated Delta recurrence geometry"));
    }
    Ok(())
}
