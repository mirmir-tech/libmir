use mircuda::{DeviceBuffer, Stream, bf16};

use super::{CudaBackend, NvFp4Config, NvFp4Tensors, u8_tensor, validate_tensors};
use crate::{
    CudaTensor, DensePlanRequest, DenseRole, ExecutionPhase, Result,
    kernels::{
        NvFp4Dequant, NvFp4DequantLaunch, NvFp4Spec, NvFp4WeightOnly, NvFp4WeightOnlyLaunch,
        NvFp4WeightOnlyTensorCore,
    },
};

mod profile;
mod tuning;
mod validation;

#[derive(Clone, Debug)]
pub struct NvFp4WeightOnlyWeight {
    weight: DeviceBuffer<u8>,
    scales: DeviceBuffer<u8>,
    global_scale: DeviceBuffer<f32>,
    materialized: CudaTensor,
    config: NvFp4Config,
}

#[derive(Debug)]
pub struct NvFp4WeightOnlyBf16Linear {
    compressed: NvFp4WeightOnly,
    tensor_core: NvFp4WeightOnlyTensorCore,
    materialized: super::super::Bf16Projection,
    stream: Stream,
    weight: NvFp4WeightOnlyWeight,
    tuning: tuning::Selection,
}

impl NvFp4WeightOnlyWeight {
    pub(in crate::backend) fn load(
        backend: &CudaBackend,
        config: NvFp4Config,
        tensors: NvFp4Tensors<'_>,
    ) -> Result<Self> {
        let spec = NvFp4Spec::new(config.input_features, config.output_features)?;
        validate_tensors(spec, tensors)?;
        let weight = u8_tensor(tensors.weight, "U8")?.clone();
        let scales = u8_tensor(tensors.weight_scale, "F8_E4M3")?.clone();
        let global_scale = tensors
            .weight_scale_2
            .as_f32()
            .ok_or_else(|| crate::Error::DTypeMismatch {
                name: tensors.weight_scale_2.name().into(),
                expected: "F32",
            })?
            .clone();
        let mut dequantized =
            backend.inner.pool.allocate::<bf16>(&backend.inner.stream, spec.elements()?)?;
        NvFp4Dequant::compile(&backend.inner.compiler, spec)?.execute(
            &backend.inner.stream,
            &mut NvFp4DequantLaunch {
                packed: &weight,
                block_scales: &scales,
                global_scale: &global_scale,
                output: &mut dequantized,
            },
        )?;
        let materialized = CudaTensor::from_bf16(
            format!("{}#bf16", tensors.weight.name()),
            vec![config.output_features, config.input_features],
            dequantized,
        );
        Ok(Self {
            weight,
            scales,
            global_scale,
            materialized,
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
        role: DenseRole,
        weight: NvFp4WeightOnlyWeight,
    ) -> Result<Self> {
        let config = weight.config;
        let spec = NvFp4Spec::new(config.input_features, config.output_features)?;
        let phase = if tokens == 1 {
            ExecutionPhase::Decode
        } else {
            ExecutionPhase::Prefill
        };
        let request = DensePlanRequest {
            phase,
            role,
            tokens,
            input_features: config.input_features,
            output_features: config.output_features,
        };
        Ok(Self {
            compressed: NvFp4WeightOnly::compile(&backend.inner.compiler, spec, tokens)?,
            tensor_core: NvFp4WeightOnlyTensorCore::compile(&backend.inner.compiler, spec, tokens)?,
            materialized: backend.prepare_bf16_projection(request)?,
            stream: backend.inner.stream.clone(),
            tuning: tuning::Selection::new(backend, tokens, config)?,
            weight,
        })
    }

    pub fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.tuning.select(
            &self.stream,
            &self.compressed,
            &self.tensor_core,
            &mut self.materialized,
            &self.weight,
            input,
            output,
        )?;
        match self.tuning.execution() {
            tuning::Execution::Compressed => self.compressed.execute(
                &self.stream,
                &mut NvFp4WeightOnlyLaunch {
                    input,
                    weight: &self.weight.weight,
                    block_scales: &self.weight.scales,
                    global_scale: &self.weight.global_scale,
                    output,
                },
            ),
            tuning::Execution::TensorCore => self.tensor_core.execute(
                &self.stream,
                &mut NvFp4WeightOnlyLaunch {
                    input,
                    weight: &self.weight.weight,
                    block_scales: &self.weight.scales,
                    global_scale: &self.weight.global_scale,
                    output,
                },
            ),
            tuning::Execution::Materialized => {
                self.materialized.execute(input, &self.weight.materialized, output)
            },
        }
    }
}
