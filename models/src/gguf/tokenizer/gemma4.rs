//! Gemma 4's `SentencePiece`-style BPE (`tokenizer.ggml.model = gemma4`) as
//! google/gemma-4-26B-A4B-it `tokenizer.json` declares it: spaces become `▁`,
//! the text splits after each `▁`, and BPE merges (kept in the GGUF file)
//! run with byte fallback.

use serde_json::{Value, json};

use super::{GgufHeader, Vocabulary, strings, vocabulary};
use crate::error::Result;

pub(super) fn tokenizer(header: &GgufHeader) -> Result<Value> {
    let Vocabulary { vocab, added, unknown, .. } = vocabulary(header)?;
    let merges = strings(header.get("tokenizer.ggml.merges"), "tokenizer.ggml.merges")?
        .into_iter()
        .map(split_merge)
        .collect::<Vec<_>>();
    Ok(json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added,
        "normalizer": {"type": "Replace", "pattern": {"String": " "}, "content": "\u{2581}"},
        "pre_tokenizer": {
            "type": "Split", "pattern": {"String": " "}, "behavior": "MergedWithPrevious",
            "invert": false,
        },
        "post_processor": null,
        "decoder": {"type": "Sequence", "decoders": [
            {"type": "Replace", "pattern": {"String": "\u{2581}"}, "content": " "},
            {"type": "ByteFallback"},
            {"type": "Fuse"},
        ]},
        "model": {
            "type": "BPE", "dropout": null, "unk_token": unknown, "continuing_subword_prefix": null,
            "end_of_word_suffix": null, "fuse_unk": true, "byte_fallback": true, "ignore_merges": false,
            "vocab": vocab,
            "merges": merges,
        },
    }))
}

/// A GGUF merge `"left right"`, split at the first space after its first
/// character as llama.cpp reads it (a piece may start with a space).
fn split_merge(merge: &str) -> Value {
    let first = merge.chars().next().map_or(0, char::len_utf8);
    merge[first..].find(' ').map_or_else(
        || json!([merge, ""]),
        |at| json!([&merge[..first + at], &merge[first + at + 1..]]),
    )
}

#[cfg(test)]
mod tests {
    use super::split_merge;

    #[test]
    fn merges_split_after_their_first_character() {
        assert_eq!(split_merge("▁t he"), serde_json::json!(["▁t", "he"]));
        // A piece that starts with a space keeps it.
        assert_eq!(split_merge("  a"), serde_json::json!([" ", "a"]));
    }
}
