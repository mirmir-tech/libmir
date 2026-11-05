//! Checks that every semantic layer or checkpoint block has tensors.

use crate::{
    error::{ModelsError, Result},
    semantic::{BlockLayoutSpec, SemanticModelSpec},
    weights::TensorCatalog,
};

pub(super) fn validate(spec: &SemanticModelSpec, catalog: &TensorCatalog) -> Result<()> {
    if let BlockLayoutSpec::Paired { physical_blocks, .. } = &spec.decoder.block_layout {
        return validate_block_coverage(*physical_blocks, catalog);
    }
    for layer in &spec.decoder.layers {
        let needle = format!("layers.{}.", layer.index);
        if !catalog.tensors.iter().any(|tensor| tensor.name.contains(&needle)) {
            return Err(ModelsError::InvalidConfig(format!(
                "checkpoint has no tensors for semantic layer {}",
                layer.index
            )));
        }
    }
    Ok(())
}

/// Every checkpoint block of a paired layout must hold at least one tensor.
fn validate_block_coverage(blocks: usize, catalog: &TensorCatalog) -> Result<()> {
    let mut present = vec![false; blocks];
    for tensor in &catalog.tensors {
        let block = ["language_model.backbone.layers.", "backbone.layers."]
            .into_iter()
            .find_map(|prefix| tensor.name.strip_prefix(prefix))
            .and_then(|remainder| remainder.split_once('.'))
            .and_then(|(block, _)| block.parse::<usize>().ok());
        if let Some(slot) = block.and_then(|block| present.get_mut(block)) {
            *slot = true;
        }
    }
    present.iter().position(|present| !present).map_or(Ok(()), |block| {
        Err(ModelsError::InvalidConfig(format!(
            "checkpoint has no tensors for block {block}"
        )))
    })
}
