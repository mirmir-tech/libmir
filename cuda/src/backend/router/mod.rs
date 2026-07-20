use mircuda::{DeviceBuffer, Stream, bf16};

mod affine;
pub use affine::AffineRouterBf16;

use super::{CudaBackend, linear::Bf16Fp32Linear};
use crate::{
    CudaTensor, Error, Result,
    kernels::{RouterSpec, RouterTopK},
};

#[cfg(test)]
mod tests;

#[derive(Clone, Copy)]
pub struct RouterTensors<'a> {
    pub projection: &'a CudaTensor,
    pub norm_scale: &'a CudaTensor,
    pub expert_scale: &'a CudaTensor,
}

#[derive(Clone, Copy)]
pub struct RouterSelection<'a> {
    pub indices: &'a DeviceBuffer<u32>,
    pub weights: &'a DeviceBuffer<bf16>,
}

#[derive(Debug)]
pub struct RouterBf16 {
    operation: RouterTopK,
    projection: Bf16Fp32Linear,
    stream: Stream,
    normalized: DeviceBuffer<bf16>,
    scores: DeviceBuffer<f32>,
    selected: DeviceBuffer<u32>,
    weights: DeviceBuffer<bf16>,
    spec: RouterSpec,
    tokens: usize,
}

impl CudaBackend {
    pub fn prepare_router_bf16(&self, spec: RouterSpec) -> Result<RouterBf16> {
        RouterBf16::new(self, spec, 1)
    }

    pub(in crate::backend) fn prepare_router_batch_bf16(
        &self,
        spec: RouterSpec,
        tokens: usize,
    ) -> Result<RouterBf16> {
        RouterBf16::new(self, spec, tokens)
    }
}

impl RouterBf16 {
    fn new(backend: &CudaBackend, spec: RouterSpec, tokens: usize) -> Result<Self> {
        let selections = tokens
            .checked_mul(spec.top_k)
            .ok_or(Error::InvalidRouter("router batch size overflow"))?;
        let normalized_elements = tokens
            .checked_mul(spec.hidden)
            .ok_or(Error::InvalidRouter("router normalized size overflow"))?;
        let score_elements = tokens
            .checked_mul(spec.experts)
            .ok_or(Error::InvalidRouter("router score size overflow"))?;
        if tokens == 0 {
            return Err(Error::InvalidRouter("router batch is empty"));
        }
        Ok(Self {
            operation: RouterTopK::compile(&backend.inner.compiler, spec)?,
            projection: Bf16Fp32Linear::new(backend, tokens, spec.hidden, spec.experts)?,
            stream: backend.inner.stream.clone(),
            normalized: backend.inner.pool.allocate(&backend.inner.stream, normalized_elements)?,
            scores: backend.inner.pool.allocate(&backend.inner.stream, score_elements)?,
            selected: backend.inner.pool.allocate::<u32>(&backend.inner.stream, selections)?,
            weights: backend.inner.pool.allocate::<bf16>(&backend.inner.stream, selections)?,
            spec,
            tokens,
        })
    }

    pub fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        tensors: RouterTensors<'_>,
    ) -> Result<RouterSelection<'_>> {
        validate(tensors.projection, &[self.spec.experts, self.spec.hidden])?;
        validate(tensors.norm_scale, &[self.spec.hidden])?;
        validate(tensors.expert_scale, &[self.spec.experts])?;
        self.operation.normalize(
            &self.stream,
            input,
            bf16(tensors.norm_scale)?,
            &mut self.normalized,
            self.tokens,
        )?;
        self.projection
            .execute(&self.normalized, tensors.projection, &mut self.scores)?;
        self.operation.select(
            &self.stream,
            &self.scores,
            bf16(tensors.expert_scale)?,
            &mut self.selected,
            &mut self.weights,
            self.tokens,
        )?;
        Ok(RouterSelection {
            indices: &self.selected,
            weights: &self.weights,
        })
    }
}

fn validate(tensor: &CudaTensor, expected: &[usize]) -> Result<()> {
    if tensor.shape() == expected {
        Ok(())
    } else {
        Err(Error::InvalidQuantizedTensor {
            name: tensor.name().into(),
            expected: expected.to_vec(),
            actual: tensor.shape().to_vec(),
        })
    }
}

fn bf16(tensor: &CudaTensor) -> Result<&DeviceBuffer<bf16>> {
    tensor.as_bf16().ok_or_else(|| Error::DTypeMismatch {
        name: tensor.name().into(),
        expected: "BF16",
    })
}
