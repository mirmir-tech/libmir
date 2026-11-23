//! Qwen3-VL and Qwen3.5 vision towers (`qwen3vl_merger`), as llama.cpp's
//! converter (`conversion/qwen.py`, `Qwen3VLVisionModel`) writes them: the
//! checkpoint's tensors under `v.` and `mm.` names, except the temporal patch
//! embedding `[out, channels, 2, patch, patch]`, split into one tensor per
//! frame (`v.patch_embd.weight`, `v.patch_embd.weight.1`).

use std::path::Path;

use serde_json::{Map, Value, json};

use super::{VisionImport, payload, safetensors, token_id};
use crate::{
    error::{ModelsError, Result},
    gguf::{GgufHeader, MetadataValue},
};

const PREFIX: &str = "model.visual";
const PATCH_EMBEDDING: &str = "model.visual.patch_embed.proj.weight";
const ASSEMBLED: &str = "mmproj-patch-embedding.safetensors";

/// Block roles: llama.cpp name, checkpoint name.
const BLOCK: [(&str, &str); 6] = [
    ("attn_qkv", "attn.qkv"),
    ("attn_out", "attn.proj"),
    ("ffn_up", "mlp.linear_fc1"),
    ("ffn_down", "mlp.linear_fc2"),
    ("ln1", "norm1"),
    ("ln2", "norm2"),
];

const GLOBAL: [(&str, &str); 7] = [
    ("mm.0.weight", "merger.linear_fc1.weight"),
    ("mm.0.bias", "merger.linear_fc1.bias"),
    ("mm.2.weight", "merger.linear_fc2.weight"),
    ("mm.2.bias", "merger.linear_fc2.bias"),
    ("v.post_ln.weight", "merger.norm.weight"),
    ("v.post_ln.bias", "merger.norm.bias"),
    ("v.position_embd.weight", "pos_embed.weight"),
];

pub(super) fn tensor_name(gguf: &str) -> Option<String> {
    if gguf == "v.patch_embd.bias" {
        return Some(format!("{PREFIX}.patch_embed.proj.bias"));
    }
    if let Some((_, name)) = GLOBAL.iter().find(|(from, _)| *from == gguf) {
        return Some(format!("{PREFIX}.{name}"));
    }
    let (layer, rest) = gguf.strip_prefix("v.blk.")?.split_once('.')?;
    layer.bytes().all(|byte| byte.is_ascii_digit()).then_some(())?;
    let (role, suffix) = rest.rsplit_once('.')?;
    let (_, name) = BLOCK.iter().find(|(from, _)| *from == role)?;
    Some(format!("{PREFIX}.blocks.{layer}.{name}.{suffix}"))
}

pub(super) fn derive(
    header: &GgufHeader,
    file: &Path,
    text: &GgufHeader,
    text_config: &Value,
) -> Result<VisionImport> {
    let unsigned = |name: &str| {
        header.unsigned(&format!("clip.vision.{name}")).ok_or_else(|| {
            ModelsError::InvalidGguf(format!("mmproj clip.vision.{name} is missing"))
        })
    };
    let deepstack = header.get("clip.vision.is_deepstack_layers").and_then(MetadataValue::as_array);
    if deepstack.is_some_and(|layers| layers.contains(&MetadataValue::Bool(true))) {
        return Err(ModelsError::InvalidGguf("deepstack vision layers are not supported".into()));
    }
    let (frame, dtype, first) = payload(header, file, "v.patch_embd.weight")?;
    let (second_frame, second_dtype, second) = payload(header, file, "v.patch_embd.weight.1")?;
    let [out, channels, height, width] = *frame.as_slice() else {
        return Err(ModelsError::InvalidGguf("mmproj patch embedding is not 4-D".into()));
    };
    if second_frame != frame || second_dtype != dtype || height != width {
        return Err(ModelsError::InvalidGguf("mmproj patch embedding frames differ".into()));
    }
    // `[out, channels, 2, patch, patch]`: each (out, channel) plane of the
    // first frame, then the second's.
    let plane = first.len() / (out * channels);
    let mut joined = Vec::with_capacity(first.len() * 2);
    for (a, b) in first.chunks_exact(plane).zip(second.chunks_exact(plane)) {
        joined.extend_from_slice(a);
        joined.extend_from_slice(b);
    }
    let shape = vec![out, channels, 2, height, width];
    let assembled = safetensors::file(&[(PATCH_EMBEDDING, &dtype, shape, joined)])?;
    let positions = header
        .catalog(file)?
        .get("v.position_embd.weight")
        .and_then(|tensor| tensor.shape.first().copied())
        .ok_or_else(|| ModelsError::InvalidGguf("mmproj position embedding is missing".into()))?;
    let triplet = |name: &str| -> Result<Value> {
        let values = header
            .get(&format!("clip.vision.{name}"))
            .and_then(MetadataValue::as_array)
            .map(|values| values.iter().filter_map(MetadataValue::as_f64).collect::<Vec<_>>())
            .filter(|values| values.len() == 3)
            .ok_or_else(|| ModelsError::InvalidGguf(format!("mmproj {name} is not 3 values")))?;
        Ok(json!(values))
    };
    let (patch, merge) = (unsigned("patch_size")?, unsigned("spatial_merge_size")?);
    let text_type = text_config.get("model_type").and_then(Value::as_str).unwrap_or_default();
    let model_type = text_type.strip_suffix("_text").unwrap_or(text_type);
    let mut root = Map::new();
    root.insert("model_type".into(), json!(model_type));
    root.insert(
        "vision_config".into(),
        json!({
            "model_type": model_type,
            "depth": unsigned("block_count")?,
            "hidden_size": unsigned("embedding_length")?,
            "intermediate_size": unsigned("feed_forward_length")?,
            "num_heads": unsigned("attention.head_count")?,
            "in_channels": channels,
            "patch_size": patch,
            "temporal_patch_size": 2,
            "spatial_merge_size": merge,
            "num_position_embeddings": positions,
            "out_hidden_size": unsigned("projection_dim")?,
            "hidden_act": "gelu_pytorch_tanh",
            "deepstack_visual_indexes": [],
        }),
    );
    for (key, token) in [
        ("image_token_id", "<|image_pad|>"),
        ("video_token_id", "<|video_pad|>"),
        ("vision_start_token_id", "<|vision_start|>"),
        ("vision_end_token_id", "<|vision_end|>"),
    ] {
        root.insert(key.into(), json!(token_id(text, token)?));
    }
    Ok(VisionImport {
        root,
        // Qwen3-VL's processor bounds (`Qwen2VLImageProcessorFast`): 256 to
        // 16384 patches of 16 pixels.
        preprocessor_config: json!({
            "size": {"shortest_edge": 65_536, "longest_edge": 16_777_216},
            "patch_size": patch,
            "temporal_patch_size": 2,
            "merge_size": merge,
            "image_mean": triplet("image_mean")?,
            "image_std": triplet("image_std")?,
            "processor_class": "Qwen3VLProcessor",
            "image_processor_type": "Qwen2VLImageProcessorFast",
        }),
        assembled: Some((ASSEMBLED.into(), assembled)),
    })
}
