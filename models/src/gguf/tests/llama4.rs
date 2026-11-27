//! Llama 4's chunked attention as llama.cpp runs it.

use super::{TestResult, Writer};
use crate::gguf::{GgufHeader, GgufModel, config::config};

fn chunk(window: Option<u32>) -> TestResult<serde_json::Value> {
    let mut writer = Writer::new().text("general.architecture", "llama4")?.kv(
        "llama4.block_count",
        4,
        4_u32.to_le_bytes().to_vec(),
    );
    if let Some(window) = window {
        writer = writer.kv("llama4.attention.sliding_window", 4, window.to_le_bytes().to_vec());
    }
    let header = GgufHeader::parse(&writer.bytes()?, None).map_err(|error| format!("{error:?}"))?;
    let config = config(&GgufModel::from_headers(vec![("model.gguf".into(), header)], [])?)?;
    Ok(config["attention_chunk_size"].clone())
}

/// llama.cpp (`llama_model_llama4::load_arch_hparams`) reads
/// `attention.sliding_window` only to tell a zero, no chunking, and
/// otherwise chunks by 8192 tokens, the size its temperature tuning is
/// fixed to: a file declaring another window runs as 8192 there, so it
/// imports as 8192 here.
#[test]
fn llama4_chunks_as_llama_cpp_does() -> TestResult {
    assert_eq!(chunk(None)?, 8192);
    assert_eq!(chunk(Some(0))?, 0);
    assert_eq!(chunk(Some(4096))?, 8192);
    Ok(())
}
