use std::sync::{Arc, Mutex};

use models::weights::{
    BlockProjectionLayout, BlockQuantization, RoutedDecoderLayerBindings, RoutedExpertBindings,
    TensorBinding, TensorStorage,
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
    let (gate_up_scales, gate_up_bias) = block_companions(
        gate_up,
        BlockProjectionLayout::FusedGateUpBank {
            experts: config.experts,
            interleaved: true,
        },
    )?;
    let (down_scales, down_bias) =
        block_companions(down, BlockProjectionLayout::MatrixBank { matrices: config.experts })?;
    let experts = ClampedRoutedExpertWeights::Native(Box::new(NativeExpertWeights {
        gate_up_blocks: tensor(tensors, &gate_up.source)?,
        gate_up_scales: tensor(tensors, gate_up_scales)?,
        gate_up_bias: tensor(tensors, gate_up_bias)?,
        down_blocks: tensor(tensors, &down.source)?,
        down_scales: tensor(tensors, down_scales)?,
        down_bias: tensor(tensors, down_bias)?,
        marlin: Arc::new(Mutex::new(None)),
    }));
    validate_native_experts(config, &experts)?;
    Ok((
        ClampedRoutedQkvProjections::Native(backend.pack_bf16_linears([&q, &k, &v])?),
        ClampedRoutedLinearWeight::Native(tensor(tensors, &bindings.attention_output.source)?),
        ClampedRoutedLinearWeight::Native(tensor(tensors, &bindings.router.source)?),
        experts,
    ))
}

fn block_companions(
    binding: &TensorBinding,
    expected: BlockProjectionLayout,
) -> Result<(&str, &str)> {
    let TensorStorage::BlockQuantized {
        format: BlockQuantization::MXFP4,
        scales,
        bias: Some(bias),
        ..
    } = &binding.storage
    else {
        return Err(crate::Error::InvalidDecoderKernel(
            "native clamped-routed expert binding requires MXFP4 scales and bias",
        ));
    };
    if binding.block_projection_layout() != Some(expected) {
        return Err(crate::Error::InvalidDecoderKernel(
            "native clamped-routed expert binding has the wrong matrix-bank layout",
        ));
    }
    Ok((scales, bias))
}

#[cfg(test)]
mod tests {
    use models::weights::{
        BindingTransform, ExpertProjectionRole, LayerTensorRole, LogicalTensorRole, TensorPacking,
    };

    use super::*;

    #[test]
    fn requires_typed_interleaved_expert_bank() -> Result<()> {
        let valid = binding(TensorPacking::InterleavedGateUp);
        let expected = BlockProjectionLayout::FusedGateUpBank { experts: 8, interleaved: true };
        assert_eq!(block_companions(&valid, expected)?, ("scales", "bias"));

        let invalid = binding(TensorPacking::Separate);
        assert!(block_companions(&invalid, expected).is_err());
        Ok(())
    }

    fn binding(packing: TensorPacking) -> TensorBinding {
        TensorBinding {
            role: LogicalTensorRole::Layer {
                index: 0,
                tensor: LayerTensorRole::ExpertProjection {
                    expert: None,
                    projection: ExpertProjectionRole::GateUp,
                },
            },
            source: "blocks".into(),
            shape: vec![8, 64, 1, 16],
            logical_shape: Some(vec![8, 64, 32]),
            transforms: vec![
                BindingTransform::StackedExperts { count: 8 },
                BindingTransform::FusedGateUp { interleaved: true },
            ],
            storage: TensorStorage::BlockQuantized {
                format: BlockQuantization::MXFP4,
                scales: "scales".into(),
                global_scale: None,
                input_scale: None,
                bias: Some("bias".into()),
                packing,
            },
        }
    }
}
