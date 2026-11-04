use mircuda::{DeviceBuffer, Stream, bf16};

use super::{NativeNvFp4Plan, NativeNvFp4Weight, NvFp4Preparation, scale_elements};
use crate::{CudaBackend, Error, Result, kernels::NvFp4RmsNorm};

#[derive(Debug)]
pub(in crate::backend::linear::nvfp4) struct NativeNvFp4Pack<const N: usize> {
    scratch: PackScratch<N>,
    weights: [NativeNvFp4Weight; N],
}

/// Plans and activation buffers of a pack; independent of which layer's
/// weights of the same geometry run through them.
#[derive(Debug)]
struct PackScratch<const N: usize> {
    plans: [NativeNvFp4Plan; N],
    preparation: NvFp4Preparation,
    rms_norm: NvFp4RmsNorm,
    stream: Stream,
    input: DeviceBuffer<u8>,
    input_scales: DeviceBuffer<u8>,
    norm_inverse: DeviceBuffer<f32>,
    tokens: usize,
    input_features: usize,
}

impl<const N: usize> NativeNvFp4Pack<N> {
    pub(in crate::backend::linear::nvfp4) fn new(
        backend: &CudaBackend,
        tokens: usize,
        weights: [NativeNvFp4Weight; N],
    ) -> Result<Self> {
        let first = weights.first().ok_or(Error::InvalidNvFp4("projection pack is empty"))?;
        let input_features = first.config.input_features;
        if weights.iter().any(|weight| {
            weight.config.input_features != input_features
                || weight.input_global.to_bits() != first.input_global.to_bits()
        }) {
            return Err(Error::InvalidNvFp4("projection pack input geometry or scale differs"));
        }
        let plans = weights
            .iter()
            .map(|weight| {
                NativeNvFp4Plan::new(
                    &backend.inner.context,
                    &backend.inner.stream,
                    tokens,
                    weight.config.output_features,
                    input_features,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let Ok(plans) = plans.try_into() else {
            return Err(Error::InvalidNvFp4("projection pack plan count differs"));
        };
        let elements = tokens
            .checked_mul(input_features)
            .ok_or(Error::InvalidNvFp4("projection pack input overflow"))?;
        let scratch = PackScratch {
            plans,
            preparation: NvFp4Preparation::compile(&backend.inner.compiler)?,
            rms_norm: NvFp4RmsNorm::compile(&backend.inner.compiler, tokens, input_features)?,
            stream: backend.inner.stream.clone(),
            input: backend.inner.pool.allocate(&backend.inner.stream, elements / 2)?,
            input_scales: backend
                .inner
                .pool
                .allocate_zeroed(&backend.inner.stream, scale_elements(tokens, input_features)?)?,
            norm_inverse: backend.inner.pool.allocate(&backend.inner.stream, tokens)?,
            tokens,
            input_features,
        };
        Ok(Self { scratch, weights })
    }

    pub(in crate::backend::linear::nvfp4) fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        outputs: &mut [DeviceBuffer<bf16>; N],
    ) -> Result<()> {
        self.scratch.execute(self.weights.each_ref(), input, outputs)
    }

    pub(in crate::backend::linear::nvfp4) fn execute_rms_norm(
        &mut self,
        input: &DeviceBuffer<bf16>,
        norm_weight: &DeviceBuffer<bf16>,
        epsilon: f32,
        outputs: &mut [DeviceBuffer<bf16>; N],
    ) -> Result<()> {
        self.scratch
            .execute_rms_norm(self.weights.each_ref(), input, norm_weight, epsilon, outputs)
    }

    /// [`Self::execute_rms_norm`] with another layer's weights of the same
    /// geometry, so one executor serves every layer.
    pub(in crate::backend::linear::nvfp4) fn execute_rms_norm_with(
        &mut self,
        weights: [&NativeNvFp4Weight; N],
        input: &DeviceBuffer<bf16>,
        norm_weight: &DeviceBuffer<bf16>,
        epsilon: f32,
        outputs: &mut [DeviceBuffer<bf16>; N],
    ) -> Result<()> {
        self.require_geometry(weights)?;
        self.scratch.execute_rms_norm(weights, input, norm_weight, epsilon, outputs)
    }

    fn require_geometry(&self, weights: [&NativeNvFp4Weight; N]) -> Result<()> {
        let first = weights[0].input_global.to_bits();
        let same = weights.iter().zip(&self.weights).all(|(weight, planned)| {
            weight.config == planned.config && weight.input_global.to_bits() == first
        });
        if same {
            Ok(())
        } else {
            Err(Error::InvalidNvFp4("layer weights differ from the projection pack geometry"))
        }
    }
}

impl<const N: usize> PackScratch<N> {
    fn execute(
        &mut self,
        weights: [&NativeNvFp4Weight; N],
        input: &DeviceBuffer<bf16>,
        outputs: &mut [DeviceBuffer<bf16>; N],
    ) -> Result<()> {
        self.preparation.quantize(
            &self.stream,
            self.tokens,
            self.input_features,
            input,
            &weights[0].input_scale,
            &mut self.input,
            &mut self.input_scales,
        )?;
        self.execute_plans(weights, outputs)
    }

    fn execute_rms_norm(
        &mut self,
        weights: [&NativeNvFp4Weight; N],
        input: &DeviceBuffer<bf16>,
        norm_weight: &DeviceBuffer<bf16>,
        epsilon: f32,
        outputs: &mut [DeviceBuffer<bf16>; N],
    ) -> Result<()> {
        self.rms_norm.execute(
            &self.stream,
            input,
            norm_weight,
            &weights[0].input_scale,
            &mut self.norm_inverse,
            &mut self.input,
            &mut self.input_scales,
            epsilon,
        )?;
        self.execute_plans(weights, outputs)
    }

    fn execute_plans(
        &mut self,
        weights: [&NativeNvFp4Weight; N],
        outputs: &mut [DeviceBuffer<bf16>; N],
    ) -> Result<()> {
        for ((plan, weight), output) in self.plans.iter_mut().zip(weights).zip(outputs.iter_mut()) {
            plan.execute(
                &self.stream,
                &self.input,
                &self.input_scales,
                &weight.weight,
                &weight.weight_scales,
                output,
                weight.alpha,
            )?;
        }
        Ok(())
    }
}
