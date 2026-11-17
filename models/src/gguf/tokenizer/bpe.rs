use serde_json::{Value, json};

use super::{GgufHeader, Vocabulary, strings, vocabulary};
use crate::error::{ModelsError, Result};

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
        // Llama 3: unsloth/Llama-3.1-8B-Instruct `tokenizer.json` (revision
        // 4699cc7), llama.cpp v1.0.0 `LLAMA_VOCAB_PRE_TYPE_LLAMA3`.
        "llama-bpe" => Some(PreTokenizer {
            pattern: r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            nfc: false,
            ignore_merges: true,
        }),
        _ => None,
    }
}

/// `tokenizer.json` for byte-level BPE vocabularies (`gpt2`).
pub(super) fn tokenizer(header: &GgufHeader) -> Result<Value> {
    let pre_name = header.string("tokenizer.ggml.pre").unwrap_or("");
    let pre = pre_tokenizer(pre_name).ok_or_else(|| {
        ModelsError::InvalidGguf(format!("pre-tokenizer `{pre_name}` has no declared pattern"))
    })?;
    let merges = strings(header.get("tokenizer.ggml.merges"), "tokenizer.ggml.merges")?;
    let Vocabulary { vocab, added, .. } = vocabulary(header)?;
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
