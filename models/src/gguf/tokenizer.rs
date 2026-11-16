//! The `tokenizer.json` a GGUF model's vocabulary declares.
//!
//! GGUF stores the vocabulary, merges and token types, and names its
//! pre-tokenizer by `tokenizer.ggml.pre`. What that name stands for (split
//! pattern, normalization) is an import default with its source recorded,
//! like the defaults keyed by `model_type`; an undeclared name is refused.

use serde_json::{Map, Value, json};

use super::{GgufModel, MetadataValue};
use crate::error::{ModelsError, Result};

/// GGUF token types (`gguf-py` `TokenType`).
const NORMAL: i64 = 1;
const CONTROL: i64 = 3;
const USER_DEFINED: i64 = 4;
const UNUSED: i64 = 5;
const BYTE: i64 = 6;

/// A byte-level BPE pre-tokenizer as the source model's `tokenizer.json`
/// declares it.
struct PreTokenizer {
    pattern: &'static str,
    nfc: bool,
    ignore_merges: bool,
}

/// `tokenizer.ggml.pre` → the source tokenizer. `qwen2`: Qwen2/Qwen3
/// `tokenizer.json` (Qwen/Qwen3-8B revision b968826), which llama.cpp v1.0
/// `llama-vocab.cpp` also attributes to `qwen2`.
fn pre_tokenizer(name: &str) -> Option<PreTokenizer> {
    match name {
        "qwen2" => Some(PreTokenizer {
            pattern: r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            nfc: true,
            ignore_merges: false,
        }),
        // o200k: openai/gpt-oss-20b `tokenizer.json` (revision 6cee5e8), the
        // pattern llama.cpp v1.0.0 lists for `LLAMA_VOCAB_PRE_TYPE_GPT4O`.
        "gpt-4o" => Some(PreTokenizer {
            pattern: r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            nfc: false,
            ignore_merges: true,
        }),
        _ => None,
    }
}

/// Builds `tokenizer.json` for byte-level BPE vocabularies (`gpt2`).
pub fn tokenizer(model: &GgufModel) -> Result<Value> {
    let header = model.header();
    let kind = header.string("tokenizer.ggml.model").unwrap_or("");
    if kind != "gpt2" {
        return Err(ModelsError::InvalidGguf(format!(
            "tokenizer model `{kind}` is not yet supported"
        )));
    }
    let pre_name = header.string("tokenizer.ggml.pre").unwrap_or("");
    let pre = pre_tokenizer(pre_name).ok_or_else(|| {
        ModelsError::InvalidGguf(format!("pre-tokenizer `{pre_name}` has no declared pattern"))
    })?;
    let tokens = strings(header.get("tokenizer.ggml.tokens"), "tokenizer.ggml.tokens")?;
    let merges = strings(header.get("tokenizer.ggml.merges"), "tokenizer.ggml.merges")?;
    let types = header
        .get("tokenizer.ggml.token_type")
        .and_then(MetadataValue::as_array)
        .ok_or_else(|| ModelsError::InvalidGguf("tokenizer.ggml.token_type is missing".into()))?;
    if types.len() != tokens.len() {
        return Err(ModelsError::InvalidGguf("token types and tokens differ in length".into()));
    }
    let mut vocab = Map::new();
    let mut added = Vec::new();
    for (id, (token, kind)) in tokens.iter().zip(types).enumerate() {
        match kind.as_i64() {
            Some(NORMAL | BYTE) => {
                vocab.insert((*token).to_owned(), json!(id));
            },
            Some(kind @ (CONTROL | USER_DEFINED)) => added.push(json!({
                "id": id, "content": token, "single_word": false, "lstrip": false,
                "rstrip": false, "normalized": false, "special": kind == CONTROL,
            })),
            Some(UNUSED) => {},
            other => {
                return Err(ModelsError::InvalidGguf(format!(
                    "token {id} has unsupported type {other:?}"
                )));
            },
        }
    }
    let byte_level = json!({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false});
    Ok(json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added,
        "normalizer": if pre.nfc { json!({"type": "NFC"}) } else { Value::Null },
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": pre.pattern}, "behavior": "Isolated", "invert": false},
            byte_level,
        ]},
        "post_processor": byte_level,
        "decoder": byte_level,
        "model": {
            "type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": "",
            "end_of_word_suffix": "", "fuse_unk": false, "byte_fallback": false, "ignore_merges": pre.ignore_merges,
            "vocab": vocab,
            "merges": merges.iter().map(|merge| merge.split_once(' ').map_or_else(
                || json!([merge, ""]),
                |(left, right)| json!([left, right]),
            )).collect::<Vec<_>>(),
        },
    }))
}

/// `tokenizer_config.json` with the chat template and special tokens the
/// metadata declares.
#[must_use]
pub fn tokenizer_config(model: &GgufModel) -> Value {
    let header = model.header();
    let tokens = header.get("tokenizer.ggml.tokens").and_then(MetadataValue::as_array);
    let token = |key: &str| {
        let id = usize::try_from(header.unsigned(key)?).ok()?;
        tokens?.get(id)?.as_str().map(str::to_owned)
    };
    let mut config = Map::new();
    if let Some(template) = header.string("tokenizer.chat_template") {
        config.insert("chat_template".into(), json!(template));
    }
    for (key, field) in [
        ("tokenizer.ggml.bos_token_id", "bos_token"),
        ("tokenizer.ggml.eos_token_id", "eos_token"),
        ("tokenizer.ggml.padding_token_id", "pad_token"),
    ] {
        if let Some(text) = token(key) {
            config.insert(field.into(), json!(text));
        }
    }
    if let Some(add) = header.get("tokenizer.ggml.add_bos_token").and_then(MetadataValue::as_bool) {
        config.insert("add_bos_token".into(), json!(add));
    }
    Value::Object(config)
}

fn strings<'a>(value: Option<&'a MetadataValue>, key: &str) -> Result<Vec<&'a str>> {
    value
        .and_then(MetadataValue::as_array)
        .ok_or_else(|| ModelsError::InvalidGguf(format!("{key} is missing")))?
        .iter()
        .map(|item| {
            item.as_str()
                .ok_or_else(|| ModelsError::InvalidGguf(format!("{key} holds a non-string")))
        })
        .collect()
}
