use mircuda::{DeviceBuffer, Stream, bf16};

use super::{CudaBackend, NvFp4Config, NvFp4Tensors, u8_tensor, validate_tensors};
use crate::{
    Result,
    kernels::{NvFp4Spec, NvFp4WeightOnly, NvFp4WeightOnlyLaunch},
};

#[derive(Clone, Debug)]
pub struct NvFp4WeightOnlyWeight {
    weight: DeviceBuffer<u8>,
    scales: DeviceBuffer<u8>,
    global_scale: DeviceBuffer<f32>,
    config: NvFp4Config,
}

#[derive(Debug)]
pub struct NvFp4WeightOnlyBf16Linear {
    operation: NvFp4WeightOnly,
    stream: Stream,
    weight: NvFp4WeightOnlyWeight,
}

impl NvFp4WeightOnlyWeight {
    pub(in crate::backend) fn load(config: NvFp4Config, tensors: NvFp4Tensors<'_>) -> Result<Self> {
        let spec = NvFp4Spec::new(config.input_features, config.output_features)?;
        validate_tensors(spec, tensors)?;
        Ok(Self {
            weight: u8_tensor(tensors.weight, "U8")?.clone(),
            scales: u8_tensor(tensors.weight_scale, "F8_E4M3")?.clone(),
            global_scale: tensors
                .weight_scale_2
                .as_f32()
                .ok_or_else(|| crate::Error::DTypeMismatch {
                    name: tensors.weight_scale_2.name().into(),
                    expected: "F32",
                })?
                .clone(),
            config,
        })
    }

    #[must_use]
    pub const fn config(&self) -> NvFp4Config {
        self.config
    }
}

impl NvFp4WeightOnlyBf16Linear {
    pub fn new(
        backend: &CudaBackend,
        tokens: usize,
        weight: NvFp4WeightOnlyWeight,
    ) -> Result<Self> {
        let spec = NvFp4Spec::new(weight.config.input_features, weight.config.output_features)?;
        Ok(Self {
            operation: NvFp4WeightOnly::compile(&backend.inner.compiler, spec, tokens)?,
            stream: backend.inner.stream.clone(),
            weight,
        })
    }

    pub fn execute(
        &self,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.operation.execute(
            &self.stream,
            &mut NvFp4WeightOnlyLaunch {
                input,
                weight: &self.weight.weight,
                block_scales: &self.weight.scales,
                global_scale: &self.weight.global_scale,
                output,
            },
        )
    }
}
