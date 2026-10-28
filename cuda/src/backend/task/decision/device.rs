use mircuda::{DeviceBuffer, DeviceElement, Stream};

use super::{
    plans::{Input, Plans},
    weights::{DeviceLinear, DeviceNorm, Write},
};
use crate::{
    CudaBackend, Result,
    kernels::{DecisionAttention, DecisionElementwise, DecisionLayout},
};

/// Kernels, plans and backend every forward step needs.
pub struct Device<'a> {
    pub backend: &'a CudaBackend,
    pub elementwise: &'a DecisionElementwise,
    pub attention: &'a DecisionAttention,
    pub layout: &'a DecisionLayout,
    pub plans: &'a Plans,
}

impl Device<'_> {
    pub fn stream(&self) -> &Stream {
        &self.backend.inner.stream
    }

    pub fn buffer<T: DeviceElement>(&self, elements: usize) -> Result<DeviceBuffer<T>> {
        Ok(self.backend.inner.pool.allocate(self.stream(), elements)?)
    }

    pub fn norm<I: Input>(
        &self,
        input: &DeviceBuffer<f32>,
        norm: &DeviceNorm,
    ) -> Result<DeviceBuffer<I>> {
        let mut output = self.buffer(input.len())?;
        let parameters = (&norm.weight, &norm.bias, norm.epsilon);
        I::norm(self.layout, self.stream(), input, parameters, &mut output)?;
        Ok(output)
    }

    pub fn linear<I: Input>(
        &self,
        linear: &DeviceLinear<I>,
        input: &DeviceBuffer<I>,
    ) -> Result<DeviceBuffer<f32>> {
        let mut output = self.buffer(input.len() / linear.inputs * linear.outputs)?;
        linear.apply(self, input, &mut output, Write::Overwrite)?;
        Ok(output)
    }
}
