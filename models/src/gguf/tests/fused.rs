//! Fused Phi projections split back into the checkpoint's projections.

use std::path::Path;

use super::{TestResult, Writer, parse};
use crate::gguf::GgmlType;

#[test]
fn phi3_fused_rows_read_as_separate_projections() -> TestResult {
    let mut writer = Writer::new().text("general.architecture", "phi3")?;
    for (key, value) in [
        ("embedding_length", 64_u32),
        ("attention.head_count", 4),
        ("attention.head_count_kv", 2),
        ("feed_forward_length", 96),
    ] {
        writer = writer.kv(&format!("phi3.{key}"), 4, value.to_le_bytes().to_vec());
    }
    // [q 64 | k 32 | v 32] and [gate 96 | up 96] rows of 64 Q8_0 values.
    let bytes = writer
        .tensor("blk.0.attn_qkv.weight", &[64, 128], GgmlType::Q8_0)?
        .tensor("blk.0.ffn_up.weight", &[64, 192], GgmlType::Q8_0)?
        .bytes()?;
    let header = parse(&bytes).map_err(|error| format!("{error:?}"))?;
    let catalog = header.mapped_catalog(Path::new("model.gguf"))?;
    let row = 2 * 34;
    let get = |name: &str| catalog.get(name).ok_or_else(|| format!("{name} is missing"));
    let (query, key, value) = (
        get("model.layers.0.self_attn.q_proj.weight")?,
        get("model.layers.0.self_attn.k_proj.weight")?,
        get("model.layers.0.self_attn.v_proj.weight")?,
    );
    assert_eq!((query.shape.clone(), key.shape.clone()), (vec![64, 64], vec![32, 64]));
    assert_eq!(key.data_offsets[0], query.data_offsets[1]);
    assert_eq!(value.data_offsets[1] - value.data_offsets[0], 32 * row);
    let (gate, up) = (
        get("model.layers.0.mlp.gate_proj.weight")?,
        get("model.layers.0.mlp.up_proj.weight")?,
    );
    assert_eq!((gate.shape.clone(), up.shape.clone()), (vec![96, 64], vec![96, 64]));
    assert_eq!(up.data_offsets[0], gate.data_offsets[1]);
    Ok(())
}
