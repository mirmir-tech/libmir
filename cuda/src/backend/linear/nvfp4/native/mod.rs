use mircuda::{DeviceBuffer, Stream, bf16};

use super::{CudaBackend, NvFp4Config, NvFp4Tensors, output_elements};
use crate::{
    Error, Result,
    kernels::{NvFp4Gated, NvFp4Preparation, scale_elements},
};

mod pack;
mod plan;
mod weight;
pub(super) use pack::NativeNvFp4Pack;
use plan::NativeNvFp4Plan;
pub(super) use weight::NativeNvFp4Weight;

#[derive(Debug)]
pub(super) struct NativeNvFp4 {
    plan: NativeNvFp4Plan,
    preparation: NvFp4Preparation,
    gated: NvFp4Gated,
    stream: Stream,
    weight: NativeNvFp4Weight,
    input: DeviceBuffer<u8>,
    input_scales: DeviceBuffer<u8>,
    tokens: usize,
}

impl NativeNvFp4 {
    pub(super) fn new(
        backend: &CudaBackend,
        tokens: usize,
        config: NvFp4Config,
        tensors: NvFp4Tensors<'_>,
    ) -> Result<Self> {
        Self::from_weight(backend, tokens, NativeNvFp4Weight::prepare(backend, config, tensors)?)
    }

    pub(super) fn from_weight(
        backend: &CudaBackend,
        tokens: usize,
        weight: NativeNvFp4Weight,
    ) -> Result<Self> {
        let config = weight.config;
        let preparation = NvFp4Preparation::compile(&backend.inner.compiler)?;
        let gated = NvFp4Gated::compile(&backend.inner.compiler, tokens, config.input_features)?;
        let input_elements = tokens
            .checked_mul(config.input_features)
            .ok_or(Error::InvalidNvFp4("input layout overflow"))?;
        let input = backend.inner.pool.allocate::<u8>(&backend.inner.stream, input_elements / 2)?;
        let input_scales = backend.inner.pool.allocate_zeroed::<u8>(
            &backend.inner.stream,
            scale_elements(tokens, config.input_features)?,
        )?;
        let plan = NativeNvFp4Plan::new(
            &backend.inner.context,
            &backend.inner.stream,
            tokens,
            config.output_features,
            config.input_features,
        )?;
        Ok(Self {
            plan,
            preparation,
            gated,
            stream: backend.inner.stream.clone(),
            weight,
            input,
            input_scales,
            tokens,
        })
    }

    pub(super) fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let Self {
            plan,
            preparation,
            stream,
            weight,
            input: quantized,
            input_scales,
            tokens,
            ..
        } = self;
        quantize(preparation, stream, *tokens, weight, input, quantized, input_scales)?;
        plan.execute(
            stream,
            quantized,
            input_scales,
            &weight.weight,
            &weight.weight_scales,
            output,
            weight.alpha,
        )
    }

    /// Runs this projection's plan with another layer's weight of the same
    /// geometry, so one executor serves every layer.
    pub(super) fn execute_with(
        &mut self,
        weight: &NativeNvFp4Weight,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.require_geometry(weight)?;
        let Self {
            plan,
            preparation,
            stream,
            input: quantized,
            input_scales,
            tokens,
            ..
        } = self;
        quantize(preparation, stream, *tokens, weight, input, quantized, input_scales)?;
        plan.execute(
            stream,
            quantized,
            input_scales,
            &weight.weight,
            &weight.weight_scales,
            output,
            weight.alpha,
        )
    }

    /// [`Self::execute_gated`] with another layer's weight of the same
    /// geometry.
    pub(super) fn execute_gated(
        &mut self,
        weight: &NativeNvFp4Weight,
        gate: &DeviceBuffer<bf16>,
        up: &DeviceBuffer<bf16>,
        activation: crate::kernels::GatedActivation,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.require_geometry(weight)?;
        let Self {
            plan, gated, stream, input, input_scales, ..
        } = self;
        gated.execute(stream, gate, up, &weight.input_scale, input, input_scales, activation)?;
        plan.execute(
            stream,
            input,
            input_scales,
            &weight.weight,
            &weight.weight_scales,
            output,
            weight.alpha,
        )
    }

    fn require_geometry(&self, weight: &NativeNvFp4Weight) -> Result<()> {
        if weight.config == self.weight.config {
            Ok(())
        } else {
            Err(Error::InvalidNvFp4("layer weight differs from the projection plan geometry"))
        }
    }

    pub(super) fn output_elements(&self) -> Result<usize> {
        output_elements(self.tokens, self.weight.config.output_features)
    }
}

fn quantize(
    preparation: &NvFp4Preparation,
    stream: &Stream,
    tokens: usize,
    weight: &NativeNvFp4Weight,
    input: &DeviceBuffer<bf16>,
    quantized: &mut DeviceBuffer<u8>,
    scales: &mut DeviceBuffer<u8>,
) -> Result<()> {
    preparation.quantize(
        stream,
        tokens,
        weight.config.input_features,
        input,
        &weight.input_scale,
        quantized,
        scales,
    )
}
