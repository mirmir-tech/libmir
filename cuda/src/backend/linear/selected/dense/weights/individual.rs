use mircuda::{DeviceBuffer, bf16};
use models::weights::{
    BindingTransform, ExpertProjectionRole, RoutedExpertBindings, TensorStorage,
};

use super::{DenseGateUpWeights, DenseProjectionWeight, required, validate};
use crate::{
    CudaBackend, CudaTensor, CudaTensorSet, Error, Result, kernels::DenseExpertInterleave,
};

/// Checkpoints with one tensor per expert and projection (Hugging Face
/// Qwen3-MoE) stack into `[experts, out, in]` banks at load; gate and up
/// interleave into one fused bank for the tensor-core expert GEMM.
pub(super) fn stack(
    backend: &CudaBackend,
    tensors: &CudaTensorSet,
    bindings: RoutedExpertBindings<'_>,
    experts: usize,
    hidden: usize,
    intermediate: usize,
) -> Result<(DenseGateUpWeights, DenseProjectionWeight)> {
    let gate = bank(
        backend,
        tensors,
        bindings,
        ExpertProjectionRole::Gate,
        experts,
        [intermediate, hidden],
    )?;
    let up = bank(
        backend,
        tensors,
        bindings,
        ExpertProjectionRole::Up,
        experts,
        [intermediate, hidden],
    )?;
    let down = bank(
        backend,
        tensors,
        bindings,
        ExpertProjectionRole::Down,
        experts,
        [hidden, intermediate],
    )?;
    let mut fused = backend.inner.pool.allocate(&backend.inner.stream, gate.len() * 2)?;
    DenseExpertInterleave::compile(&backend.inner.compiler)?.execute(
        &backend.inner.stream,
        &gate,
        &up,
        &mut fused,
        experts * intermediate,
        hidden,
    )?;
    let plain = |name: &str, shape, weight| DenseProjectionWeight {
        weight: CudaTensor::from_bf16(name.into(), shape, weight),
        bias: None,
        transposed: false,
    };
    Ok((
        DenseGateUpWeights::Fused {
            projection: plain(
                "experts.gate_up#stacked",
                vec![experts, 2 * intermediate, hidden],
                fused,
            ),
            interleaved: true,
        },
        plain("experts.down#stacked", vec![experts, hidden, intermediate], down),
    ))
}

fn bank(
    backend: &CudaBackend,
    tensors: &CudaTensorSet,
    bindings: RoutedExpertBindings<'_>,
    projection: ExpertProjectionRole,
    experts: usize,
    [output, input]: [usize; 2],
) -> Result<DeviceBuffer<bf16>> {
    let parts = bindings.individual(projection);
    if parts.len() != experts {
        return Err(Error::UnsupportedDecoderLayer(format!(
            "checkpoint binds {} of {experts} {projection:?} experts",
            parts.len()
        )));
    }
    let matrix = output * input;
    let mut bank = backend.inner.pool.allocate(&backend.inner.stream, experts * matrix)?;
    for (expert, binding) in parts.into_iter().enumerate() {
        if !matches!(binding.storage, TensorStorage::Dense { bias: None, .. })
            || binding.transforms.contains(&BindingTransform::Transpose)
        {
            return Err(Error::UnsupportedDecoderLayer(format!(
                "stacked CUDA experts need dense untransposed tensors without bias: {}",
                binding.source
            )));
        }
        let tensor = required(tensors, &binding.source)?;
        validate(&tensor, &[output, input])?;
        let source = tensor.as_bf16().ok_or_else(|| Error::DTypeMismatch {
            name: binding.source.clone(),
            expected: "BF16",
        })?;
        backend
            .inner
            .stream
            .copy_device_range(source, 0..matrix, &mut bank, expert * matrix)?;
    }
    Ok(bank)
}
