pub(super) use architecture::CudaClampedExpertLayout as ClampedRoutedLayout;
use models::weights::WeightBindingPlan;

use crate::{Error, Result};

pub(super) fn discover(bindings: &WeightBindingPlan) -> Result<ClampedRoutedLayout> {
    ClampedRoutedLayout::discover(bindings).ok_or_else(|| {
        Error::UnsupportedDecoderLayer(
            "binding plan has no complete clamped-routed expert projection layout".into(),
        )
    })
}

#[cfg(test)]
mod tests {
    use models::weights::{
        AffineBits, AffineGroupAxis, AffinePacking, AffineParameterDType, AffineSignedness,
        AffineStorageDType, AffineZeroPointMode, BindingTransform, BlockQuantization,
        ExpertProjectionRole, GroupedAffineQuantization, LayerTensorRole, LogicalTensorRole,
        TensorBinding, TensorPacking, TensorStorage, WeightBindingPlan,
    };

    use super::*;

    #[test]
    fn distinguishes_native_and_mlx_checkpoints() -> Result<()> {
        let native = plan(vec![binding(
            ExpertProjectionRole::GateUp,
            TensorStorage::BlockQuantized {
                format: BlockQuantization::MXFP4,
                scales: "scales".into(),
                global_scale: None,
                input_scale: None,
                bias: None,
                packing: TensorPacking::InterleavedGateUp,
            },
        )]);
        assert_eq!(discover(&native)?, ClampedRoutedLayout::Native);

        let affine = || TensorStorage::AffineQuantized {
            format: GroupedAffineQuantization {
                bits: AffineBits::Four,
                group_size: 32,
                group_axis: AffineGroupAxis::Input,
                signedness: AffineSignedness::Unsigned,
                zero_point: AffineZeroPointMode::None,
                packing: AffinePacking::Mlx,
                storage_dtype: AffineStorageDType::U32,
                scale_dtype: AffineParameterDType::F16,
                bias_dtype: None,
            },
            scales: "scales".into(),
            biases: None,
            output_bias: None,
        };
        let mlx = plan(vec![
            binding(ExpertProjectionRole::Gate, affine()),
            binding(ExpertProjectionRole::Up, affine()),
        ]);
        assert_eq!(discover(&mlx)?, ClampedRoutedLayout::Mlx);

        let dense = plan(vec![binding(
            ExpertProjectionRole::GateUp,
            TensorStorage::Dense {
                dtype: models::weights::DenseDType::Bf16,
                bias: None,
            },
        )]);
        assert_eq!(discover(&dense)?, ClampedRoutedLayout::Dense);
        Ok(())
    }

    #[test]
    fn rejects_a_plan_without_an_expert_transform() {
        assert!(discover(&plan(Vec::new())).is_err());
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
