//! Gemma 4 vision towers (`gemma4v`), as llama.cpp's converter
//! (`conversion/gemma.py`, `Gemma4VisionAudioModel`) writes them: the
//! checkpoint's tensors under `v.` and `mm.` names, the patch projection
//! reshaped to `[out, channels, patch, patch]` from the checkpoint's
//! `[out, patch · patch · channels]` rows, which the import restores. The
//! tower's `RoPE` base, pooling kernel and soft-token budget are the
//! constants llama.cpp's `mtmd` and the checkpoint's processor declare.

use std::path::Path;

use serde_json::{Map, Value, json};

use super::{VisionImport, payload, safetensors, token_id};
use crate::{
    error::{ModelsError, Result},
    gguf::{GgufHeader, MetadataValue},
};

const PREFIX: &str = "model.vision_tower";
const PATCH_PROJECTION: &str = "model.vision_tower.patch_embedder.input_proj.weight";
const ASSEMBLED: &str = "mmproj-patch-projection.safetensors";
/// `pooling_kernel_size` unless `clip.vision.projector.scale_factor` says.
const POOLING_KERNEL: u64 = 3;
const SOFT_TOKENS: u64 = 280;
const ROPE_THETA: f64 = 100.0;

const BLOCK: [(&str, &str); 13] = [
    ("attn_q", "self_attn.q_proj.linear"),
    ("attn_k", "self_attn.k_proj.linear"),
    ("attn_v", "self_attn.v_proj.linear"),
    ("attn_out", "self_attn.o_proj.linear"),
    ("attn_q_norm", "self_attn.q_norm"),
    ("attn_k_norm", "self_attn.k_norm"),
    ("ln1", "input_layernorm"),
    ("ln2", "pre_feedforward_layernorm"),
    ("attn_post_norm", "post_attention_layernorm"),
    ("ffn_post_norm", "post_feedforward_layernorm"),
    ("ffn_gate", "mlp.gate_proj.linear"),
    ("ffn_up", "mlp.up_proj.linear"),
    ("ffn_down", "mlp.down_proj.linear"),
];

/// The scalar bounds a clipped linear clamps its input and output to.
/// The clip bound whose presence marks a clipped tower.
const INPUT_MAX: &str = "input_max";
const CLIP_BOUNDS: [&str; 4] = ["input_min", INPUT_MAX, "output_min", "output_max"];

const GLOBAL: [(&str, &str); 4] = [
    (
        "v.position_embd.weight",
        "model.vision_tower.patch_embedder.position_embedding_table",
    ),
    ("v.std_bias", "model.vision_tower.std_bias"),
    ("v.std_scale", "model.vision_tower.std_scale"),
    ("mm.input_projection.weight", "model.embed_vision.embedding_projection.weight"),
];

pub(super) fn tensor_name(gguf: &str) -> Option<String> {
    if let Some((_, name)) = GLOBAL.iter().find(|(from, _)| *from == gguf) {
        return Some((*name).to_owned());
    }
    let (layer, rest) = gguf.strip_prefix("v.blk.")?.split_once('.')?;
    layer.bytes().all(|byte| byte.is_ascii_digit()).then_some(())?;
    let (role, suffix) = rest.rsplit_once('.')?;
    let (_, name) = BLOCK.iter().find(|(from, _)| *from == role)?;
    // A clipped linear's bounds sit beside its `linear` module.
    let name = if CLIP_BOUNDS.contains(&suffix) {
        name.strip_suffix(".linear")?
    } else {
        name
    };
    Some(format!("{PREFIX}.encoder.layers.{layer}.{name}.{suffix}"))
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
    let catalog = header.catalog(file)?;
    let clipped = catalog.tensors.iter().any(|tensor| {
        tensor.name.starts_with("v.blk.")
            && tensor.name.rsplit_once('.').is_some_and(|(_, leaf)| leaf == INPUT_MAX)
    });
    let (hidden, heads) = (unsigned("embedding_length")?, unsigned("attention.head_count")?);
    let (patch, channels) = (unsigned("patch_size")?, 3_u64);
    let assembled = patch_projection(header, file, channels)?;
    let positions = catalog
        .get("v.position_embd.weight")
        .and_then(|tensor| tensor.shape.get(1).copied())
        .ok_or_else(|| ModelsError::InvalidGguf("mmproj position table is missing".into()))?;
    let kernel = header.unsigned("clip.vision.projector.scale_factor").unwrap_or(POOLING_KERNEL);
    let epsilon = header
        .get("clip.vision.attention.layer_norm_epsilon")
        .and_then(MetadataValue::as_f64)
        .map_or(1e-6, |value| format!("{value:.6e}").parse().unwrap_or(1e-6));
    let triplet = |name: &str| -> Result<Value> {
        header
            .get(&format!("clip.vision.{name}"))
            .and_then(MetadataValue::as_array)
            .map(|values| {
                json!(values.iter().filter_map(MetadataValue::as_f64).collect::<Vec<_>>())
            })
            .ok_or_else(|| ModelsError::InvalidGguf(format!("mmproj {name} is missing")))
    };
    let mut root = Map::new();
    root.insert("model_type".into(), json!("gemma4"));
    root.insert(
        "vision_config".into(),
        json!({
            "model_type": "gemma4_vision",
            "hidden_size": hidden,
            "intermediate_size": unsigned("feed_forward_length")?,
            "num_hidden_layers": unsigned("block_count")?,
            "num_attention_heads": heads,
            "num_key_value_heads": heads,
            "head_dim": hidden / heads.max(1),
            "patch_size": patch,
            "pooling_kernel_size": kernel,
            "position_embedding_size": positions,
            "rms_norm_eps": epsilon,
            "hidden_activation": "gelu_pytorch_tanh",
            "rope_parameters": {"rope_type": "default", "rope_theta": ROPE_THETA},
            "standardize": catalog.contains("v.std_bias"),
            "use_clipped_linears": clipped,
            "default_output_length": SOFT_TOKENS,
        }),
    );
    for (key, token) in [
        ("image_token_id", "<|image|>"),
        ("boi_token_id", "<|image>"),
        ("eoi_token_id", "<image|>"),
    ] {
        root.insert(key.into(), json!(token_id(text, token)?));
    }
    root.insert("vision_soft_tokens_per_image".into(), json!(SOFT_TOKENS));
    // Image tokens attend to each other in both directions.
    let mut text_config = text_config.clone();
    text_config["use_bidirectional_attention"] = json!("vision");
    root.insert("text_config".into(), text_config);
    Ok(VisionImport {
        root,
        preprocessor_config: json!({"image_processor": {
            "image_processor_type": "Gemma4ImageProcessor",
            "patch_size": patch,
            "pooling_kernel_size": kernel,
            "max_soft_tokens": SOFT_TOKENS,
            "image_seq_length": SOFT_TOKENS,
            "do_resize": true,
            "do_rescale": true,
            "rescale_factor": 1.0 / 255.0,
            "do_normalize": false,
            "image_mean": triplet("image_mean")?,
            "image_std": triplet("image_std")?,
        }}),
        assembled: Some((ASSEMBLED.into(), assembled)),
    })
}

/// The patch projection `[out, channels, h, w]` back as the checkpoint's
/// `[out, h · w · channels]` rows, in a `SafeTensors` file.
fn patch_projection(header: &GgufHeader, file: &Path, channels: u64) -> Result<Vec<u8>> {
    let (shape, dtype, bytes) = payload(header, file, "v.patch_embd.weight")?;
    let [out, stored_channels, height, width] = *shape.as_slice() else {
        return Err(ModelsError::InvalidGguf("mmproj patch projection is not 4-D".into()));
    };
    if u64::try_from(stored_channels)? != channels || height != width {
        return Err(ModelsError::InvalidGguf("mmproj patch projection is not RGB patches".into()));
    }
    // `[out, channels, h, w]` back to the checkpoint's `[out, h · w ·
    // channels]` rows.
    let element = bytes.len() / (out * stored_channels * height * width);
    let mut rows = Vec::with_capacity(bytes.len());
    for o in 0..out {
        for y in 0..height {
            for x in 0..width {
                for c in 0..stored_channels {
                    let at = (((o * stored_channels + c) * height + y) * width + x) * element;
                    rows.extend_from_slice(&bytes[at..at + element]);
                }
            }
        }
    }
    let assembled = safetensors::file(&[(
        PATCH_PROJECTION,
        &dtype,
        vec![out, height * width * stored_channels],
        rows,
    )])?;
    Ok(assembled)
}
