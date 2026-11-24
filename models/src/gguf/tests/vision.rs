//! `mmproj` vision towers imported beside their text model.

use super::{TestResult, Writer, parse};
use crate::{
    gguf::{GgmlType, GgufModel, Projector, VisionImport},
    weights::TensorCatalog,
};

fn unsigned(value: u32) -> Vec<u8> {
    value.to_le_bytes().to_vec()
}

fn array<T>(kind: u32, values: &[T], item: impl Fn(&T) -> Vec<u8>) -> Vec<u8> {
    let mut bytes = unsigned(kind);
    bytes.extend(u64::try_from(values.len()).unwrap_or_default().to_le_bytes());
    bytes.extend(values.iter().flat_map(item));
    bytes
}

/// A two-block `qwen3vl_merger` tower with `[4, 3, 2, 2]` frames.
pub(super) fn mmproj() -> TestResult<Writer> {
    let mut writer = Writer::new()
        .text("general.architecture", "clip")?
        .text("clip.projector_type", "qwen3vl_merger")?;
    for (key, value) in [
        ("block_count", 2),
        ("embedding_length", 4),
        ("feed_forward_length", 8),
        ("attention.head_count", 2),
        ("patch_size", 2),
        ("spatial_merge_size", 2),
        ("projection_dim", 6),
    ] {
        writer = writer.kv(&format!("clip.vision.{key}"), 4, unsigned(value));
    }
    for key in ["image_mean", "image_std"] {
        writer = writer.kv(
            &format!("clip.vision.{key}"),
            9,
            array(6, &[0.5_f32; 3], |value| value.to_le_bytes().to_vec()),
        );
    }
    Ok(writer
        .tensor("v.patch_embd.weight", &[2, 2, 3, 4], GgmlType::F32)?
        .tensor("v.patch_embd.weight.1", &[2, 2, 3, 4], GgmlType::F32)?
        .tensor("v.position_embd.weight", &[4, 16], GgmlType::F32)?
        .tensor("v.blk.1.attn_qkv.weight", &[4, 12], GgmlType::F16)?)
}

fn text() -> TestResult<Writer> {
    let tokens = ["a", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>"];
    Ok(Writer::new().text("general.architecture", "qwen35")?.kv(
        "tokenizer.ggml.tokens",
        9,
        array(8, &tokens, |token| Writer::string(token).unwrap_or_default()),
    ))
}

/// The `mmproj` file sorts before the text model; the directory's model is
/// still the text one.
#[test]
fn the_text_model_is_not_the_projector_beside_it() -> TestResult {
    let directory = std::env::temp_dir().join(format!("libmir-gguf-text-{}", std::process::id()));
    std::fs::create_dir_all(&directory)?;
    std::fs::write(directory.join("mmproj-F16.gguf"), mmproj()?.bytes()?)?;
    std::fs::write(directory.join("model-Q4_K_M.gguf"), text()?.bytes()?)?;
    let model = crate::gguf::text_model(&directory);
    std::fs::remove_dir_all(&directory)?;
    let architecture = model?.map(|model| model.header().architecture().map(str::to_owned));
    assert_eq!(architecture, Some(Some("qwen35".to_owned())));
    Ok(())
}

#[test]
fn names_follow_the_checkpoint() -> TestResult {
    let header = parse(&mmproj()?.bytes()?).map_err(|error| format!("{error:?}"))?;
    assert_eq!(Projector::of(&header)?, Some(Projector::Qwen3VlMerger));
    let catalog = header.mapped_catalog(std::path::Path::new("mmproj.gguf"))?;
    for name in ["model.visual.blocks.1.attn.qkv.weight", "model.visual.pos_embed.weight"] {
        assert!(catalog.contains(name), "{name} is missing");
    }
    // The split frames stay unmapped; the import assembles them.
    assert!(catalog.contains("v.patch_embd.weight.1"));
    Ok(())
}

#[test]
fn the_import_joins_the_patch_frames() -> TestResult {
    let directory = std::env::temp_dir().join(format!("libmir-mmproj-{}", std::process::id()));
    std::fs::create_dir_all(&directory)?;
    let (vision, model) = (directory.join("mmproj.gguf"), directory.join("text.gguf"));
    let mut bytes = mmproj()?.bytes()?;
    let header = parse(&bytes).map_err(|error| format!("{error:?}"))?;
    // Frame 0 holds 1.0 and frame 1 holds 2.0 in every value.
    for (index, tensor) in header.tensors.iter().take(2).enumerate() {
        let start = usize::try_from(header.data_start + tensor.offset)?;
        let value = f32::from(u8::try_from(index + 1)?).to_le_bytes();
        for chunk in bytes[start..start + 48 * 4].as_chunks_mut::<4>().0 {
            *chunk = value;
        }
    }
    std::fs::write(&vision, &bytes)?;
    std::fs::write(&model, text()?.bytes()?)?;
    let config = serde_json::json!({"model_type": "qwen3_5_text"});
    let import =
        VisionImport::derive(&GgufModel::open(&vision)?, &GgufModel::open(&model)?, &config);
    std::fs::remove_dir_all(&directory)?;
    let import = import?;
    assert_eq!(import.root["model_type"], "qwen3_5");
    assert_eq!(import.root["vision_config"]["num_position_embeddings"], 16);
    assert_eq!(
        (
            import.root["image_token_id"].clone(),
            import.root["vision_end_token_id"].clone()
        ),
        (3.into(), 2.into())
    );
    let (_, file) = import.assembled.ok_or("nothing assembled")?;
    let catalog = TensorCatalog::from_safetensors_header("patch.safetensors", &file)?;
    let weight = catalog.get("model.visual.patch_embed.proj.weight").ok_or("weight is missing")?;
    assert_eq!(weight.shape, [4, 3, 2, 2, 2]);
    let data = &file[usize::try_from(weight.payload_start()?)?..];
    let values = data
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect::<Vec<_>>();
    // Each (out, channel) plane of four values: frame 0, then frame 1.
    assert_eq!(&values[..8], &[1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]);
    assert_eq!(values.len(), 96);
    Ok(())
}

/// A one-block `gemma4v` tower with one `[2, 3, 2, 2]` patch projection.
fn gemma4v() -> TestResult<Writer> {
    let mut writer = Writer::new()
        .text("general.architecture", "clip")?
        .text("clip.vision.projector_type", "gemma4v")?;
    for (key, value) in [
        ("block_count", 1),
        ("embedding_length", 4),
        ("feed_forward_length", 8),
        ("attention.head_count", 2),
        ("patch_size", 2),
    ] {
        writer = writer.kv(&format!("clip.vision.{key}"), 4, unsigned(value));
    }
    for (key, value) in [("image_mean", 0.0_f32), ("image_std", 1.0)] {
        writer = writer.kv(
            &format!("clip.vision.{key}"),
            9,
            array(6, &[value; 3], |value| value.to_le_bytes().to_vec()),
        );
    }
    Ok(writer
        .tensor("v.patch_embd.weight", &[2, 2, 3, 2], GgmlType::F32)?
        .tensor("v.position_embd.weight", &[4, 6, 2], GgmlType::F32)?
        .tensor("v.blk.0.attn_out.weight", &[4, 4], GgmlType::F16)?)
}

#[test]
fn gemma4_patch_rows_return_to_pixel_major_order() -> TestResult {
    let directory = std::env::temp_dir().join(format!("libmir-gemma4v-{}", std::process::id()));
    std::fs::create_dir_all(&directory)?;
    let (vision, model) = (directory.join("mmproj.gguf"), directory.join("text.gguf"));
    let mut bytes = gemma4v()?.bytes()?;
    let header = parse(&bytes).map_err(|error| format!("{error:?}"))?;
    let catalog = header.mapped_catalog(std::path::Path::new("mmproj.gguf"))?;
    assert!(catalog.contains("model.vision_tower.encoder.layers.0.self_attn.o_proj.linear.weight"));
    // Value at [out, channel, y, x] is `channel * 10 + y * 2 + x + out * 100`.
    let start = usize::try_from(header.data_start + header.tensors[0].offset)?;
    for (index, chunk) in bytes[start..start + 24 * 4].as_chunks_mut::<4>().0.iter_mut().enumerate()
    {
        let (out, channel, pixel) = (index / 12, index / 4 % 3, index % 4);
        *chunk = f32::from(u16::try_from(out * 100 + channel * 10 + pixel)?).to_le_bytes();
    }
    std::fs::write(&vision, &bytes)?;
    let tokens = ["<|image>", "<|image|>", "<image|>"];
    let text = Writer::new().text("general.architecture", "gemma4")?.kv(
        "tokenizer.ggml.tokens",
        9,
        array(8, &tokens, |token| Writer::string(token).unwrap_or_default()),
    );
    std::fs::write(&model, text.bytes()?)?;
    let config = serde_json::json!({"model_type": "gemma4_text"});
    let import =
        VisionImport::derive(&GgufModel::open(&vision)?, &GgufModel::open(&model)?, &config);
    std::fs::remove_dir_all(&directory)?;
    let import = import?;
    assert_eq!(import.root["vision_config"]["position_embedding_size"], 6);
    assert_eq!(import.root["image_token_id"], 1);
    assert_eq!(import.root["text_config"]["use_bidirectional_attention"], "vision");
    let (_, file) = import.assembled.ok_or("nothing assembled")?;
    let catalog = TensorCatalog::from_safetensors_header("patch.safetensors", &file)?;
    let weight = catalog
        .get("model.vision_tower.patch_embedder.input_proj.weight")
        .ok_or("projection is missing")?;
    assert_eq!(weight.shape, [2, 12]);
    let data = &file[usize::try_from(weight.payload_start()?)?..];
    let values = data
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect::<Vec<_>>();
    // Row 0: pixel (0,0) channels 0, 1, 2, then pixel (0,1).
    assert_eq!(&values[..6], &[0.0, 10.0, 20.0, 1.0, 11.0, 21.0]);
    Ok(())
}
