use std::{collections::BTreeSet, path::PathBuf};

use super::*;

#[test]
fn mxfp8_hint_distinguishes_block_weights_from_affine_int8() -> Result<()> {
    let catalog = TensorCatalog {
        tensors: vec![
            tensor("projection.weight", "U32", vec![16, 8]),
            tensor("projection.scales", "U8", vec![16, 1]),
        ],
    };
    let mut consumed = BTreeSet::new();

    let storage =
        block::mlx_storage("projection", &catalog, &mut consumed, Some(BlockQuantization::MXFP8))
            .ok_or_else(|| ModelsError::InvalidConfig("missing MXFP8 storage".into()))?;

    assert!(matches!(
        storage,
        TensorStorage::BlockQuantized { format: BlockQuantization::MXFP8, .. }
    ));
    assert!(consumed.contains("projection.scales"));
    Ok(())
}

#[test]
fn mxfp4_hint_preserves_u32_container_and_affine_overrides() -> Result<()> {
    let catalog = TensorCatalog {
        tensors: vec![
            tensor("expert.weight", "U32", vec![16, 8]),
            tensor("expert.scales", "U8", vec![16, 1]),
            tensor("router.weight", "U32", vec![16, 8]),
            tensor("router.scales", "BF16", vec![16, 1]),
            tensor("router.biases", "BF16", vec![16, 1]),
        ],
    };
    let mut consumed = BTreeSet::new();
    let storage =
        block::mlx_storage("expert", &catalog, &mut consumed, Some(BlockQuantization::MXFP4_MLX))
            .ok_or_else(|| ModelsError::InvalidConfig("missing MXFP4 storage".into()))?;
    assert!(matches!(
        storage,
        TensorStorage::BlockQuantized { format: BlockQuantization::MXFP4_MLX, .. }
    ));
    assert!(
        block::mlx_storage("router", &catalog, &mut consumed, Some(BlockQuantization::MXFP4_MLX),)
            .is_none()
    );
    Ok(())
}

fn tensor(name: &str, dtype: &str, shape: Vec<usize>) -> TensorInfo {
    TensorInfo {
        name: name.into(),
        file: PathBuf::new(),
        dtype: dtype.into(),
        shape,
        data_start: 0,
        data_offsets: [0, 0],
    }
}
