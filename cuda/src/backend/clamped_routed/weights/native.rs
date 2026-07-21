use models::weights::{
    RoutedDecoderLayerBindings, RoutedExpertBindings, TensorBinding, TensorStorage,
};

use super::{ClampedRoutedConfig, ClampedRoutedExpertWeights, NativeExpertWeights, tensor};
use crate::{
    CudaBackend, CudaTensorSet, Result,
    backend::clamped_routed::{
        projection::{ClampedRoutedLinearWeight, ClampedRoutedQkvProjections},
        validation::validate_native_experts,
    },
};

pub(super) fn load(
    backend: &CudaBackend,
    config: ClampedRoutedConfig,
    tensors: &CudaTensorSet,
    bindings: RoutedDecoderLayerBindings<'_>,
) -> Result<(
    ClampedRoutedQkvProjections,
    ClampedRoutedLinearWeight,
    ClampedRoutedLinearWeight,
    ClampedRoutedExpertWeights,
)> {
    let q = tensor(tensors, &bindings.query.source)?;
    let k = tensor(tensors, &bindings.key.source)?;
    let v = tensor(tensors, &bindings.value.source)?;
    let RoutedExpertBindings::InterleavedGateUp { gate_up, down } = bindings.experts else {
        return Err(crate::Error::InvalidDecoderKernel(
            "native clamped-routed requires interleaved gate/up expert bindings",
        ));
    };
    let (gate_up_scales, gate_up_bias) = block_companions(gate_up)?;
    let (down_scales, down_bias) = block_companions(down)?;
    let experts = ClampedRoutedExpertWeights::Native(Box::new(NativeExpertWeights {
        gate_up_blocks: tensor(tensors, &gate_up.source)?,
        gate_up_scales: tensor(tensors, gate_up_scales)?,
        gate_up_bias: tensor(tensors, gate_up_bias)?,
        down_blocks: tensor(tensors, &down.source)?,
        down_scales: tensor(tensors, down_scales)?,
        down_bias: tensor(tensors, down_bias)?,
    }));
    validate_native_experts(config, &experts)?;
    Ok((
        ClampedRoutedQkvProjections::Native(backend.pack_bf16_linears([&q, &k, &v])?),
        ClampedRoutedLinearWeight::Native(tensor(tensors, &bindings.attention_output.source)?),
        ClampedRoutedLinearWeight::Native(tensor(tensors, &bindings.router.source)?),
        experts,
    ))
}

fn block_companions(binding: &TensorBinding) -> Result<(&str, &str)> {
    let TensorStorage::BlockQuantized { scales, bias: Some(bias), .. } = &binding.storage else {
        return Err(crate::Error::InvalidDecoderKernel(
            "native clamped-routed expert binding requires block scales and bias",
        ));
    };
    Ok((scales, bias))
}
