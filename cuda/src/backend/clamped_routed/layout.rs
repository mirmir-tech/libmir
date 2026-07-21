use models::weights::{ExpertProjectionLayout, WeightBindingPlan};

use crate::{Error, Result};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ClampedRoutedLayout {
    Native,
    Mlx,
}

impl ClampedRoutedLayout {
    pub(super) fn discover(bindings: &WeightBindingPlan) -> Result<Self> {
        match bindings.expert_projection_layout() {
            Some(ExpertProjectionLayout::InterleavedGateUp) => Ok(Self::Native),
            Some(ExpertProjectionLayout::SeparateGateUp) => Ok(Self::Mlx),
            None => Err(Error::UnsupportedDecoderLayer(
                "binding plan has no complete clamped-routed expert projection layout".into(),
            )),
        }
    }

    pub(super) const fn storage(self) -> &'static str {
        match self {
            Self::Native => "interleaved MXFP4 gate/up blocks",
            Self::Mlx => "separate MLX affine MXFP4 gate/up blocks",
        }
    }
}

#[cfg(test)]
mod tests {
    use models::weights::{
        BindingTransform, BlockFormat, ExpertProjectionRole, LayerTensorRole, LogicalTensorRole,
        TensorBinding, TensorPacking, TensorStorage, WeightBindingPlan,
    };

    use super::*;

    #[test]
    fn distinguishes_native_and_mlx_checkpoints() -> Result<()> {
        let native = plan(vec![binding(
            ExpertProjectionRole::GateUp,
            TensorStorage::BlockQuantized {
                format: BlockFormat::MxFp4,
                scales: "scales".into(),
                global_scale: None,
                input_scale: None,
                bias: None,
                packing: TensorPacking::InterleavedGateUp,
            },
        )]);
        assert_eq!(ClampedRoutedLayout::discover(&native)?, ClampedRoutedLayout::Native);

        let affine = || TensorStorage::AffineQuantized {
            dtype: "U32".into(),
            bits: Some(4),
            scales: "scales".into(),
            biases: None,
            output_bias: None,
            group_size: Some(32),
        };
        let mlx = plan(vec![
            binding(ExpertProjectionRole::Gate, affine()),
            binding(ExpertProjectionRole::Up, affine()),
        ]);
        assert_eq!(ClampedRoutedLayout::discover(&mlx)?, ClampedRoutedLayout::Mlx);
        Ok(())
    }

    #[test]
    fn rejects_a_plan_without_an_expert_transform() {
        assert!(ClampedRoutedLayout::discover(&plan(Vec::new())).is_err());
    }

    fn plan(tensors: Vec<TensorBinding>) -> WeightBindingPlan {
        WeightBindingPlan { tensors }
    }

    fn binding(projection: ExpertProjectionRole, storage: TensorStorage) -> TensorBinding {
        TensorBinding {
            role: LogicalTensorRole::Layer {
                index: 0,
                tensor: LayerTensorRole::ExpertProjection { expert: None, projection },
            },
            source: String::new(),
            shape: Vec::new(),
            logical_shape: None,
            transforms: match projection {
                ExpertProjectionRole::GateUp => {
                    vec![BindingTransform::FusedGateUp { interleaved: true }]
                },
                ExpertProjectionRole::Gate
                | ExpertProjectionRole::Up
                | ExpertProjectionRole::Down => Vec::new(),
            },
            storage,
        }
    }
}
