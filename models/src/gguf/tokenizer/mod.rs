//! The `tokenizer.json` a GGUF model's vocabulary declares.
//!
//! GGUF stores the vocabulary, merges and token types, and names its
//! pre-tokenizer by `tokenizer.ggml.pre`. What that name stands for (split
//! pattern, normalization) is an import default with its source recorded,
//! like the defaults keyed by `model_type`; an undeclared name is refused.

use serde_json::{Map, Value, json};

mod bpe;
mod gemma4;
mod spm;

use super::{GgufHeader, GgufModel, MetadataValue};
use crate::error::{ModelsError, Result};

/// GGUF token types (`gguf-py` `TokenType`).
const NORMAL: i64 = 1;
const UNKNOWN: i64 = 2;
const CONTROL: i64 = 3;
const USER_DEFINED: i64 = 4;
const UNUSED: i64 = 5;
const BYTE: i64 = 6;

/// Builds `tokenizer.json` for the vocabulary's tokenizer model.
pub fn tokenizer(model: &GgufModel) -> Result<Value> {
    let header = model.header();
    let mut tokenizer = match header.string("tokenizer.ggml.model").unwrap_or("") {
        "gpt2" => bpe::tokenizer(header),
        "llama" => spm::tokenizer(header),
        "gemma4" => gemma4::tokenizer(header),
        kind => Err(ModelsError::InvalidGguf(format!(
            "tokenizer model `{kind}` is not yet supported"
        ))),
    }?;
    // Phi-3's SentencePiece vocabularies; Phi-4 (also `phi3`) is byte-level.
    if matches!(header.architecture(), Some("phi3" | "phimoe"))
        && header.string("tokenizer.ggml.model") == Some("llama")
    {
        phi3_whitespace(&mut tokenizer);
    }
    if let Some(bos) = automatic_bos(header) {
        let post = tokenizer["post_processor"].take();
        tokenizer["post_processor"] = prepend_bos(&post, &bos);
    }
    Ok(tokenizer)
}

/// Phi-3's special tokens swallow the whitespace after them, except `<unk>`,
/// `<s>` and `<|endoftext|>`; `</s>` does too (microsoft/Phi-3.5-*
/// `tokenizer.json`, and llama.cpp's `phi-3` token attributes).
fn phi3_whitespace(tokenizer: &mut Value) {
    let Some(added) = tokenizer["added_tokens"].as_array_mut() else {
        return;
    };
    for token in added {
        let content = token["content"].as_str().unwrap_or_default();
        let special = token["special"].as_bool().unwrap_or(false);
        let kept = matches!(content, "<unk>" | "<s>" | "<|endoftext|>");
        token["rstrip"] = json!((special && !kept) || content == "</s>");
    }
}

/// The BOS token the vocabulary adds to every encoding
/// (`tokenizer.ggml.add_bos_token`), as llama.cpp does for prompts whose
/// rendered template does not begin with it (Command R).
fn automatic_bos(header: &GgufHeader) -> Option<(u64, String)> {
    let add = header.get("tokenizer.ggml.add_bos_token").and_then(MetadataValue::as_bool);
    if add != Some(true) {
        return None;
    }
    let id = header.unsigned("tokenizer.ggml.bos_token_id")?;
    let tokens = header.get("tokenizer.ggml.tokens").and_then(MetadataValue::as_array)?;
    let text = tokens.get(usize::try_from(id).ok()?)?.as_str()?;
    Some((id, text.to_owned()))
}

/// A `TemplateProcessing` that leads each sequence with BOS, after `post`.
fn prepend_bos(post: &Value, (id, text): &(u64, String)) -> Value {
    let bos = json!({ "SpecialToken": { "id": text, "type_id": 0 } });
    let sequence = |name: &str| json!({ "Sequence": { "id": name, "type_id": 0 } });
    let template = json!({
        "type": "TemplateProcessing",
        "single": [bos, sequence("A")],
        "pair": [bos, sequence("A"), bos, sequence("B")],
        "special_tokens": { text.as_str(): { "id": text, "ids": [id], "tokens": [text] } },
    });
    if post.is_null() {
        template
    } else {
        json!({ "type": "Sequence", "processors": [post, template] })
    }
}

/// The vocabulary and added tokens by GGUF token type; `UNKNOWN` joins the
/// vocabulary and is returned as the unknown token.
fn vocabulary(header: &GgufHeader) -> Result<Vocabulary<'_>> {
    let tokens = strings(header.get("tokenizer.ggml.tokens"), "tokenizer.ggml.tokens")?;
    let types = header
        .get("tokenizer.ggml.token_type")
        .and_then(MetadataValue::as_array)
        .ok_or_else(|| ModelsError::InvalidGguf("tokenizer.ggml.token_type is missing".into()))?;
    if types.len() != tokens.len() {
        return Err(ModelsError::InvalidGguf("token types and tokens differ in length".into()));
    }
    let mut vocab = Map::new();
    let mut added = Vec::new();
    let mut unknown = None;
    for (id, (token, kind)) in tokens.iter().zip(types).enumerate() {
        match kind.as_i64() {
            Some(NORMAL | BYTE) => {
                vocab.insert((*token).to_owned(), json!(id));
            },
            Some(UNKNOWN) => {
                vocab.insert((*token).to_owned(), json!(id));
                unknown = Some(*token);
                added.push(added_token(id, token, true));
            },
            Some(kind @ (CONTROL | USER_DEFINED)) => {
                added.push(added_token(id, token, kind == CONTROL));
            },
            Some(UNUSED) => {},
            other => {
                return Err(ModelsError::InvalidGguf(format!(
                    "token {id} has unsupported type {other:?}"
                )));
            },
        }
    }
    // Added tokens inside the model's id range (Tekken's first thousand
    // control tokens) are model entries too, as the source `tokenizer.json`
    // lists them; otherwise the model's ids have gaps that added ids fill.
    let model_end = vocab.values().filter_map(Value::as_u64).max().unwrap_or(0);
    for token in &added {
        if let (Some(id), Some(content)) = (token["id"].as_u64(), token["content"].as_str())
            && id < model_end
        {
            vocab.entry(content.to_owned()).or_insert_with(|| json!(id));
        }
    }
    Ok(Vocabulary { tokens, vocab, added, unknown })
}

struct Vocabulary<'a> {
    tokens: Vec<&'a str>,
    vocab: Map<String, Value>,
    added: Vec<Value>,
    unknown: Option<&'a str>,
}

fn added_token(id: usize, token: &str, special: bool) -> Value {
    json!({
        "id": id, "content": token, "single_word": false, "lstrip": false,
        "rstrip": false, "normalized": false, "special": special,
    })
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

#[cfg(test)]
mod tests;
