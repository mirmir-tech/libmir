//! Projection scales beside FP8 and NVFP4 GGUF tensors.

use std::path::Path;

use super::{TestResult, Writer, parse};
use crate::gguf::GgmlType;

#[test]
fn fp8_scales_become_checkpoint_weight_scales() -> TestResult {
    let bytes = Writer::new()
        .text("general.architecture", "llama")?
        .tensor("blk.0.ffn_up.weight", &[32, 8], GgmlType::F8_E4M3)?
        .tensor("blk.0.ffn_up.scale", &[1], GgmlType::F32)?
        .tensor("blk.0.ffn_down.weight", &[64, 8], GgmlType::NVFP4)?
        .tensor("blk.0.ffn_down.scale", &[1], GgmlType::F32)?
        .bytes()?;
    let header = parse(&bytes).map_err(|error| format!("{error:?}"))?;
    let catalog = header.mapped_catalog(Path::new("model.gguf"))?;
    for name in [
        "model.layers.0.mlp.up_proj.weight_scale",
        // GGML block types keep `.scale`, their output scale.
        "model.layers.0.mlp.down_proj.scale",
    ] {
        assert!(catalog.contains(name), "{name} is missing");
    }
    assert!(!catalog.contains("model.layers.0.mlp.up_proj.scale"));
    Ok(())
}
