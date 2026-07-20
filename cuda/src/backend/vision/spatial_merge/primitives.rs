use mircuda::{DeviceBuffer, bf16};

use super::super::super::CudaBackend;
use crate::{
    CudaTensor, Error, Result,
    kernels::{VisionElementwise, VisionElementwiseSpec},
};

pub(super) fn elementwise(
    backend: &CudaBackend,
    rows: usize,
    columns: usize,
    epsilon: f32,
) -> Result<VisionElementwise> {
    VisionElementwise::compile(
        &backend.inner.compiler,
        VisionElementwiseSpec { rows, columns, epsilon },
    )
}

pub(super) fn bf16(tensor: &CudaTensor) -> Result<&DeviceBuffer<bf16>> {
    tensor.as_bf16().ok_or_else(|| Error::DTypeMismatch {
        name: tensor.name().into(),
        expected: "BF16",
    })
}
