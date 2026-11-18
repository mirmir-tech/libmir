use models::weights::{BlockProjectionLayout, BlockQuantization, TensorBinding, TensorStorage};

use super::{ClampedRoutedConfig, ExpertLayout};
use crate::engine::{Array, Dtype, Error, ModelTensors, Result};

pub(super) fn native_layout(
    tensors: &ModelTensors,
    gate_up: &TensorBinding,
    down: &TensorBinding,
    config: ClampedRoutedConfig,
) -> Result<ExpertLayout> {
    let Some(BlockProjectionLayout::FusedGateUpBank { experts, interleaved: true }) =
        gate_up.block_projection_layout()
    else {
        return Err(invalid(gate_up, "requires an interleaved gate/up expert bank"));
    };
    let (gate_up_scales, gate_up_bias) =
        companions(gate_up, BlockProjectionLayout::FusedGateUpBank { experts, interleaved: true })?;
    let (down_scales, down_bias) =
        companions(down, BlockProjectionLayout::MatrixBank { matrices: experts })?;
    let hidden = usize::try_from(config.hidden)?;
    let intermediate = usize::try_from(config.intermediate)?;
    let gate_rows = intermediate.checked_mul(2).ok_or(Error::ShapeOverflow)?;
    let layout = ExpertLayout::Native {
        gate_up_blocks: tensors.get(&gate_up.source)?,
        gate_up_scales: tensors.get(gate_up_scales)?,
        gate_up_bias: tensors.get(gate_up_bias)?,
        down_blocks: tensors.get(&down.source)?,
        down_scales: tensors.get(down_scales)?,
        down_bias: tensors.get(down_bias)?,
    };
    validate(&layout, experts, hidden, intermediate, gate_rows, gate_up, down)?;
    Ok(layout)
}

fn companions(binding: &TensorBinding, expected: BlockProjectionLayout) -> Result<(&str, &str)> {
    let TensorStorage::BlockQuantized {
        format: BlockQuantization::MXFP4,
        scales,
        bias: Some(bias),
        ..
    } = &binding.storage
    else {
        return Err(invalid(binding, "requires MXFP4 scales and BF16 bias"));
    };
    if binding.block_projection_layout() != Some(expected) {
        return Err(invalid(binding, "matrix-bank layout differs from routed role"));
    }
    Ok((scales, bias))
}

#[allow(clippy::too_many_arguments)]
fn validate(
    layout: &ExpertLayout,
    experts: usize,
    hidden: usize,
    intermediate: usize,
    gate_rows: usize,
    gate: &TensorBinding,
    down: &TensorBinding,
) -> Result<()> {
    let ExpertLayout::Native {
        gate_up_blocks,
        gate_up_scales,
        gate_up_bias,
        down_blocks,
        down_scales,
        down_bias,
    } = layout
    else {
        return Err(Error::InvalidModel("expected native MXFP4 experts".into()));
    };
    require(gate_up_blocks, Dtype::Uint8, &[experts, gate_rows, hidden / 32, 16], gate)?;
    require(gate_up_scales, Dtype::Uint8, &[experts, gate_rows, hidden / 32], gate)?;
    require(gate_up_bias, Dtype::Bfloat16, &[experts, gate_rows], gate)?;
    require(down_blocks, Dtype::Uint8, &[experts, hidden, intermediate / 32, 16], down)?;
    require(down_scales, Dtype::Uint8, &[experts, hidden, intermediate / 32], down)?;
    require(down_bias, Dtype::Bfloat16, &[experts, hidden], down)
}

fn require(array: &Array, dtype: Dtype, shape: &[usize], binding: &TensorBinding) -> Result<()> {
    let expected = shape
        .iter()
        .copied()
        .map(i32::try_from)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if array.dtype()? == dtype && array.shape()? == expected {
        Ok(())
    } else {
        Err(invalid(binding, "MXFP4 expert companion dtype or shape differs"))
    }
}

fn invalid(binding: &TensorBinding, reason: &str) -> Error {
    Error::InvalidQuantization(format!("{}: {reason}", binding.source))
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
        assert_eq!(companions(&valid, expected)?, ("scales", "bias"));

        let invalid = binding(TensorPacking::Separate);
        assert!(companions(&invalid, expected).is_err());
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
