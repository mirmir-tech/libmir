use mircuda::{DeviceBuffer, bf16};

use super::{
    CudaBackend, NvFp4Bf16Linear, NvFp4Bf16Pack, NvFp4Config, NvFp4LinearWeight, NvFp4Tensors,
    native,
};
use crate::Result;

impl NvFp4Bf16Linear {
    pub(in crate::backend) fn new(
        backend: &CudaBackend,
        tokens: usize,
        config: NvFp4Config,
        tensors: NvFp4Tensors<'_>,
    ) -> Result<Self> {
        Ok(Self {
            execution: native::NativeNvFp4::new(backend, tokens, config, tensors)?,
        })
    }

    pub fn from_weight(
        backend: &CudaBackend,
        tokens: usize,
        weight: NvFp4LinearWeight,
    ) -> Result<Self> {
        Ok(Self {
            execution: native::NativeNvFp4::from_weight(backend, tokens, weight.prepared)?,
        })
    }

    /// Quantizes input and enqueues one native FP4 projection without host
    /// sync.
    pub fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.execution.execute(input, output)
    }

    pub fn output_elements(&self) -> Result<usize> {
        self.execution.output_elements()
    }

    /// [`Self::execute`] with another layer's weight of this geometry.
    pub(in crate::backend) fn execute_with(
        &mut self,
        weight: &NvFp4LinearWeight,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.execution.execute_with(&weight.prepared, input, output)
    }

    /// Quantizes the gated activation of `gate` and `up` and projects it with a
    /// layer's weight of this geometry.
    pub(in crate::backend) fn execute_gated(
        &mut self,
        weight: &NvFp4LinearWeight,
        gate: &DeviceBuffer<bf16>,
        up: &DeviceBuffer<bf16>,
        activation: crate::kernels::GatedActivation,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.execution.execute_gated(&weight.prepared, gate, up, activation, output)
    }
}

impl<const N: usize> NvFp4Bf16Pack<N> {
    pub fn from_weights(
        backend: &CudaBackend,
        tokens: usize,
        weights: [NvFp4LinearWeight; N],
    ) -> Result<Self> {
        Ok(Self {
            execution: native::NativeNvFp4Pack::new(
                backend,
                tokens,
                weights.map(|weight| weight.prepared),
            )?,
        })
    }

    /// Quantizes input once and enqueues every projection without host sync.
    pub fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        outputs: &mut [DeviceBuffer<bf16>; N],
    ) -> Result<()> {
        self.execution.execute(input, outputs)
    }

    /// Normalizes and quantizes input in one CUDA kernel before executing all
    /// projections.
    pub fn execute_rms_norm(
        &mut self,
        input: &DeviceBuffer<bf16>,
        norm_weight: &DeviceBuffer<bf16>,
        epsilon: f32,
        outputs: &mut [DeviceBuffer<bf16>; N],
    ) -> Result<()> {
        self.execution.execute_rms_norm(input, norm_weight, epsilon, outputs)
    }

    /// [`Self::execute_rms_norm`] with another layer's weights of this
    /// geometry.
    pub(in crate::backend) fn execute_rms_norm_with(
        &mut self,
        weights: [&NvFp4LinearWeight; N],
        input: &DeviceBuffer<bf16>,
        norm_weight: &DeviceBuffer<bf16>,
        epsilon: f32,
        outputs: &mut [DeviceBuffer<bf16>; N],
    ) -> Result<()> {
        let weights = weights.map(|weight| &weight.prepared);
        self.execution
            .execute_rms_norm_with(weights, input, norm_weight, epsilon, outputs)
    }
}
