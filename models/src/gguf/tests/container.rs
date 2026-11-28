//! A GGUF import directory read through `WeightContainer`: split text shards,
//! the vision tower's `mmproj` and the `SafeTensors` file its import assembles.

use std::fs;

use super::{GgmlType, TestResult, Writer, vision::mmproj};
use crate::{
    layout::{WeightContainer, WeightFile},
    weights::TensorCatalog,
};

fn shard(number: u32, writer: Writer) -> Writer {
    let unsigned = |value: u32| value.to_le_bytes().to_vec();
    writer.kv("split.no", 4, unsigned(number)).kv("split.count", 4, unsigned(2))
}

fn safetensors() -> Vec<u8> {
    let header = r#"{"model.visual.patch_embed.proj.weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&[0; 4]);
    bytes
}

#[test]
fn reads_each_container_of_a_gguf_import_once() -> TestResult {
    let root = std::env::temp_dir().join(format!("libmir-gguf-container-{}", std::process::id()));
    fs::create_dir_all(&root)?;
    let first = shard(0, Writer::new().text("general.architecture", "nemotron_h")?)
        .kv("split.tensors.count", 4, 2_u32.to_le_bytes().to_vec())
        .tensor("blk.0.ssm_d", &[1, 4], GgmlType::F32)?;
    let second = shard(1, Writer::new()).tensor("blk.1.ssm_a", &[1, 4], GgmlType::F32)?;
    fs::write(root.join("model-00001-of-00002.gguf"), first.bytes()?)?;
    fs::write(root.join("model-00002-of-00002.gguf"), second.bytes()?)?;
    fs::write(root.join("mmproj-F16.gguf"), mmproj()?.bytes()?)?;
    fs::write(root.join("mmproj-patch-embedding.safetensors"), safetensors())?;
    fs::write(root.join("config.json"), "{}")?;
    let files = |container: WeightContainer| -> TestResult<Vec<WeightFile>> {
        Ok(container
            .discover(&root)?
            .into_iter()
            .map(WeightFile::open)
            .collect::<Result<_, _>>()?)
    };
    let gguf = files(WeightContainer::Gguf)?;
    let safetensors = files(WeightContainer::SafeTensors)?;
    let catalog = TensorCatalog::from_weight_files(&[gguf.clone(), safetensors.clone()].concat());
    let text = WeightContainer::Gguf.read_catalog(&gguf);
    let vision = WeightContainer::SafeTensors.read_catalog(&safetensors);
    fs::remove_dir_all(&root)?;
    assert_eq!(gguf.len(), 3);
    assert!(gguf.iter().all(|file| file.container == WeightContainer::Gguf));
    let names = text?.into_iter().map(|tensor| tensor.name).collect::<Vec<_>>();
    for name in ["backbone.layers.0.mixer.D", "backbone.layers.1.mixer.A_log"] {
        // Both shards are listed, yet the split model is read once.
        assert_eq!(names.iter().filter(|found| *found == name).count(), 1, "{name}");
    }
    assert!(names.iter().any(|name| name == "model.visual.pos_embed.weight"));
    let vision = vision?;
    assert_eq!(vision.len(), 1);
    assert_eq!(vision[0].name, "model.visual.patch_embed.proj.weight");
    let catalog = catalog?;
    assert!(catalog.get("model.visual.patch_embed.proj.weight").is_some());
    assert!(catalog.get("backbone.layers.1.mixer.A_log").is_some());
    Ok(())
}

#[test]
fn layout_from_gguf_derives_assets_in_memory_without_writing_to_disk() -> TestResult {
    let root = std::env::temp_dir().join(format!("libmir-gguf-inmem-{}", std::process::id()));
    drop(fs::remove_dir_all(&root));
    fs::create_dir_all(&root)?;
    let bytes = Writer::new()
        .text("general.architecture", "qwen3")?
        .kv("qwen3.block_count", 4, 1_u32.to_le_bytes().to_vec())
        .kv("qwen3.embedding_length", 4, 64_u32.to_le_bytes().to_vec())
        .kv("qwen3.feed_forward_length", 4, 128_u32.to_le_bytes().to_vec())
        .kv("qwen3.attention.head_count", 4, 2_u32.to_le_bytes().to_vec())
        .kv("qwen3.attention.head_count_kv", 4, 1_u32.to_le_bytes().to_vec())
        .kv("qwen3.attention.key_length", 4, 32_u32.to_le_bytes().to_vec())
        .kv("qwen3.attention.layer_norm_rms_epsilon", 6, 1e-6_f32.to_le_bytes().to_vec())
        .kv("qwen3.context_length", 4, 256_u32.to_le_bytes().to_vec())
        .text("tokenizer.ggml.model", "gpt2")?
        .text("tokenizer.ggml.pre", "qwen2")?
        .kv("tokenizer.ggml.tokens", 9, {
            let mut payload = 8_u32.to_le_bytes().to_vec();
            payload.extend(2_u64.to_le_bytes());
            for tok in ["a", "b"] {
                payload.extend((tok.len() as u64).to_le_bytes());
                payload.extend(tok.as_bytes());
            }
            payload
        })
        .kv("tokenizer.ggml.merges", 9, {
            let mut payload = 8_u32.to_le_bytes().to_vec();
            payload.extend(0_u64.to_le_bytes());
            payload
        })
        .kv("tokenizer.ggml.token_type", 9, {
            let mut payload = 5_u32.to_le_bytes().to_vec();
            payload.extend(2_u64.to_le_bytes());
            payload.extend(1_i32.to_le_bytes());
            payload.extend(1_i32.to_le_bytes());
            payload
        })
        .tensor("token_embd.weight", &[64, 2], GgmlType::Q8_0)?
        .tensor("output_norm.weight", &[64], GgmlType::F32)?
        .bytes()?;
    let path = root.join("tiny.gguf");
    fs::write(&path, bytes)?;
    let layout = crate::layout::ModelLayout::from_gguf(&path, None)?;
    let entries = fs::read_dir(&root)?.collect::<std::io::Result<Vec<_>>>()?;
    let catalog = TensorCatalog::from_layout(&layout)?;
    let config = layout.read_config()?;
    fs::remove_dir_all(&root)?;
    assert_eq!(entries.len(), 1, "in-memory GGUF layout must not write sidecar files");
    assert_eq!(config["hidden_size"], 64);
    assert!(catalog.get("model.embed_tokens.weight").is_some());
    assert!(catalog.get("model.norm.weight").is_some());
    Ok(())
}
