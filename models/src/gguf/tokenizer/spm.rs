//! `SentencePiece` BPE vocabularies (`tokenizer.ggml.model = llama`: Llama 2,
//! Mistral, Gemma) as Hugging Face `tokenizers` represents them: BPE over
//! `▁`-marked text with byte fallback. GGUF keeps each piece's score but no
//! merges; the merges come from the scores exactly as `transformers`'
//! `SentencePieceExtractor` derives them for these models' `tokenizer.json`.

use std::collections::HashMap;

use serde_json::{Value, json};

use super::{GgufHeader, Vocabulary, vocabulary};
use crate::{
    error::{ModelsError, Result},
    gguf::MetadataValue,
};

pub(super) fn tokenizer(header: &GgufHeader) -> Result<Value> {
    let Vocabulary { tokens, mut vocab, added, unknown } = vocabulary(header)?;
    // As in Hugging Face's conversion every piece is in the vocabulary,
    // control pieces too, so every derived merge names known pieces.
    for (id, token) in tokens.iter().enumerate() {
        vocab.entry((*token).to_owned()).or_insert_with(|| json!(id));
    }
    let scores = header
        .get("tokenizer.ggml.scores")
        .and_then(MetadataValue::as_array)
        .ok_or_else(|| ModelsError::InvalidGguf("tokenizer.ggml.scores is missing".into()))?
        .iter()
        .map(|score| {
            score
                .as_f64()
                .ok_or_else(|| ModelsError::InvalidGguf("a token score is not a number".into()))
        })
        .collect::<Result<Vec<_>>>()?;
    if scores.len() != tokens.len() {
        return Err(ModelsError::InvalidGguf("token scores and tokens differ in length".into()));
    }
    // llama.cpp prepends a space unless the converter recorded otherwise
    // (Gemma records `false`).
    let prefix = header
        .get("tokenizer.ggml.add_space_prefix")
        .and_then(MetadataValue::as_bool)
        .unwrap_or(true);
    let mut decoders = vec![
        json!({"type": "Replace", "pattern": {"String": "\u{2581}"}, "content": " "}),
        json!({"type": "ByteFallback"}),
        json!({"type": "Fuse"}),
    ];
    if prefix {
        decoders.push(json!({"type": "Strip", "content": " ", "start": 1, "stop": 0}));
    }
    Ok(json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added,
        "normalizer": null,
        "pre_tokenizer": {
            "type": "Metaspace", "replacement": "\u{2581}",
            "prepend_scheme": if prefix { "first" } else { "never" }, "split": false,
        },
        "post_processor": null,
        "decoder": {"type": "Sequence", "decoders": decoders},
        "model": {
            "type": "BPE", "dropout": null, "unk_token": unknown, "continuing_subword_prefix": null,
            "end_of_word_suffix": null, "fuse_unk": true, "byte_fallback": true, "ignore_merges": false,
            "vocab": vocab,
            "merges": merges(&tokens, &scores),
        },
    }))
}

/// Every split of every piece into two pieces, grouped by the piece's score
/// (highest first) and within a piece by the ids of its halves.
fn merges(tokens: &[&str], scores: &[f64]) -> Vec<Value> {
    let ids: HashMap<&str, usize> =
        tokens.iter().enumerate().map(|(id, token)| (*token, id)).collect();
    let mut merges = Vec::new();
    for (token, &score) in tokens.iter().zip(scores) {
        let mut local = token
            .char_indices()
            .skip(1)
            .filter_map(|(at, _)| {
                let (left, right) = token.split_at(at);
                Some((ids.get(left).copied()?, ids.get(right).copied()?, left, right))
            })
            .collect::<Vec<_>>();
        local.sort_by_key(|&(left, right, ..)| (left, right));
        merges.extend(local.into_iter().map(|(.., left, right)| (score, left, right)));
    }
    merges.sort_by(|left, right| right.0.total_cmp(&left.0));
    merges.into_iter().map(|(_, left, right)| json!([left, right])).collect()
}
