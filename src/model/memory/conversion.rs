//! Weights a backend keeps in another layout than the file's.

use models::{
    execution::DecoderExecutionContract,
    weights::{GgmlCompanion, LogicalTensorRole, TensorBinding, TensorStorage},
};

/// Bytes Metal adds by keeping GGML projections as MLX affine groups of 32
/// with 16-bit scales and biases: dense matrices without an output scale of a
/// type whose [`models::gguf::Affine32::repacks_on_metal`]. Embedding tables
/// and expert banks keep their GGML blocks; prediction modules count only when
/// they load.
pub(super) fn metal_affine_growth(
    execution: Option<&DecoderExecutionContract>,
    prediction: bool,
) -> u64 {
    execution
        .into_iter()
        .flat_map(|execution| &execution.bindings.tensors)
        .filter(|binding| match binding.role {
            LogicalTensorRole::Embedding => false,
            LogicalTensorRole::Prediction { .. } => prediction,
            _ => true,
        })
        .filter_map(growth)
        .fold(0_u64, u64::saturating_add)
}

fn growth(binding: &TensorBinding) -> Option<u64> {
    let TensorStorage::Ggml {
        format,
        companion: GgmlCompanion::Plain | GgmlCompanion::Bias(_),
    } = &binding.storage
    else {
        return None;
    };
    let affine = format.ggml_type.affine32().filter(|affine| affine.repacks_on_metal())?;
    let [rows, columns] = *binding.logical_shape.as_deref().unwrap_or(&binding.shape) else {
        return None;
    };
    let (rows, columns) = (u64::try_from(rows).ok()?, u64::try_from(columns).ok()?);
    let file = format.ggml_type.row_bytes(columns).ok()?.checked_mul(rows)?;
    Some(affine.bytes(rows.checked_mul(columns)?).saturating_sub(file))
}

#[cfg(test)]
mod tests {
    use models::{
        gguf::GgmlType,
        weights::{
            GgmlBlockQuantization, GgmlCompanion, LogicalTensorRole, TensorBinding, TensorStorage,
        },
    };

    use super::growth;

    fn binding(ggml_type: GgmlType, shape: Vec<usize>, scale: Option<String>) -> TensorBinding {
        TensorBinding {
            role: LogicalTensorRole::Output,
            source: "output.weight".into(),
            shape,
            logical_shape: None,
            transforms: Vec::new(),
            storage: TensorStorage::Ggml {
                format: GgmlBlockQuantization { ggml_type },
                companion: scale.map_or(GgmlCompanion::Plain, GgmlCompanion::Scale),
            },
        }
    }

    #[test]
    fn affine_groups_add_their_scales_over_the_file_blocks() {
        // Q4_K: 144 bytes per 256 values on file, 160 as affine groups.
        assert_eq!(growth(&binding(GgmlType::Q4_K, vec![4, 512], None)), Some(4 * 2 * 16));
        // Q4_1 already stores a 16-bit scale and minimum per 32 values.
        assert_eq!(growth(&binding(GgmlType::Q4_1, vec![4, 512], None)), Some(0));
        assert_eq!(growth(&binding(GgmlType::Q6_K, vec![4, 512], None)), None);
        assert_eq!(growth(&binding(GgmlType::Q4_K, vec![2, 4, 512], None)), None);
        assert_eq!(growth(&binding(GgmlType::Q4_0, vec![4, 512], Some("s".into()))), None);
        // Metal keeps 5- and 8-bit types as GGML blocks.
        assert_eq!(growth(&binding(GgmlType::Q8_0, vec![4, 512], None)), None);
    }
}
