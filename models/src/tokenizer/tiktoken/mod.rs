//! Tiktoken checkpoints (`tiktoken.model`: one base64 token and its rank per
//! line), rebuilt as an equivalent byte-level BPE: a token's rank orders the
//! merges that form it, and a pre-token that is itself a token stays whole.
//!
//! The split pattern and special tokens live in the checkpoint's Python
//! tokenizer, so each supported scheme names them explicitly.

mod pattern;
#[cfg(test)]
mod tests;

use std::{collections::HashMap, fs, path::Path};

use tokenizers::{
    AddedToken, Tokenizer,
    decoders::byte_level::ByteLevel as ByteLevelDecoder,
    models::bpe::{BPE, Vocab},
    pre_tokenizers::{
        byte_level::ByteLevel,
        sequence::Sequence,
        split::{Split, SplitPattern},
    },
};

pub use self::pattern::HAN_RUNS_CASED_WORDS;
pub(super) use self::pattern::TikTokenScheme;
use crate::error::{ModelsError, Result};

/// The byte-level BPE of `ranks_path` with `scheme`'s split pattern and
/// special tokens; `config_path` names the special tokens it declares.
pub(super) fn tokenizer_from_files(
    ranks_path: &Path,
    config_path: Option<&Path>,
) -> Result<Tokenizer> {
    let scheme = TikTokenScheme::from_config(config_path)?;
    let ranks = ranks(&fs::read_to_string(ranks_path)?)?;
    let alphabet = byte_alphabet();
    let text =
        |bytes: &[u8]| bytes.iter().map(|byte| alphabet[usize::from(*byte)]).collect::<String>();
    let mut vocab = Vocab::default();
    for (bytes, rank) in &ranks {
        let _previous = vocab.insert(text(bytes), *rank);
    }
    let merges = merges(&ranks)
        .into_iter()
        .map(|(left, right)| (text(left), text(right)))
        .collect();
    let model = BPE::builder().vocab_and_merges(vocab, merges).ignore_merges(true).build()?;
    let mut tokenizer = Tokenizer::new(model);
    let split = Split::new(
        SplitPattern::Regex(scheme.pattern().into()),
        tokenizers::SplitDelimiterBehavior::Isolated,
        false,
    )?;
    tokenizer.with_pre_tokenizer(Some(Sequence::new(vec![
        split.into(),
        ByteLevel::new(false, true, false).into(),
    ])));
    tokenizer.with_decoder(Some(ByteLevelDecoder::default()));
    let base = u32::try_from(ranks.len())?;
    let names = pattern::declared_special_tokens(config_path)?;
    let specials = (0..scheme.special_count())
        .map(|offset| {
            let id = base + offset;
            names.get(&id).cloned().unwrap_or_else(|| format!("<|reserved_token_{id}|>"))
        })
        .map(|name| AddedToken::from(name, true))
        .collect::<Vec<_>>();
    let added = tokenizer.add_special_tokens(&specials);
    if added != specials.len() {
        return Err(ModelsError::InvalidConfig(
            "tiktoken special tokens collide with its ranks".into(),
        ));
    }
    Ok(tokenizer)
}

/// Token bytes by rank; ranks must be `0..n` without gaps.
fn ranks(source: &str) -> Result<Vec<(Vec<u8>, u32)>> {
    let mut ranks = source
        .lines()
        .filter(|line| !line.is_empty())
        .map(|line| {
            let (token, rank) = line
                .split_once(' ')
                .ok_or_else(|| invalid("a tiktoken rank line has no rank"))?;
            Ok((base64(token)?, rank.parse::<u32>()?))
        })
        .collect::<Result<Vec<_>>>()?;
    ranks.sort_by_key(|(_, rank)| *rank);
    if ranks.iter().zip(0_u32..).any(|((_, rank), expected)| *rank != expected) {
        return Err(invalid("tiktoken ranks are not consecutive from zero"));
    }
    Ok(ranks)
}

/// Every split of a multi-byte token into two tokens, ordered by the rank of
/// the token they form, then by the ranks of the parts (as `transformers`'
/// `TikTokenConverter` derives them).
fn merges(ranks: &[(Vec<u8>, u32)]) -> Vec<(&[u8], &[u8])> {
    let index = ranks
        .iter()
        .map(|(bytes, rank)| (bytes.as_slice(), *rank))
        .collect::<HashMap<_, _>>();
    let mut merges = Vec::new();
    for (token, _) in ranks.iter().filter(|(token, _)| token.len() > 1) {
        let mut local = (1..token.len())
            .filter_map(|split| {
                let (left, right) = token.split_at(split);
                Some((index.get(left)?, index.get(right)?, left, right))
            })
            .collect::<Vec<_>>();
        local.sort_by_key(|(left, right, ..)| (**left, **right));
        merges.extend(local.into_iter().map(|(.., left, right)| (left, right)));
    }
    merges
}

/// GPT-2's printable stand-in for every byte, which byte-level BPE vocabularies
/// spell their tokens with.
fn byte_alphabet() -> Vec<char> {
    let printable = |byte: u32| {
        (u32::from(b'!')..=u32::from(b'~')).contains(&byte)
            || (0xA1..=0xAC).contains(&byte)
            || (0xAE..=0xFF).contains(&byte)
    };
    let mut shifted = 0;
    (0..256_u32)
        .map(|byte| {
            let code = if printable(byte) {
                byte
            } else {
                shifted += 1;
                255 + shifted
            };
            char::from_u32(code).unwrap_or(char::REPLACEMENT_CHARACTER)
        })
        .collect()
}

/// Standard padded base64, the only encoding tiktoken rank files use.
fn base64(text: &str) -> Result<Vec<u8>> {
    let value = |byte: u8| match byte {
        b'A'..=b'Z' => Some(byte - b'A'),
        b'a'..=b'z' => Some(byte - b'a' + 26),
        b'0'..=b'9' => Some(byte - b'0' + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    };
    let text = text.as_bytes();
    if !text.len().is_multiple_of(4) {
        return Err(invalid("a tiktoken token is not padded base64"));
    }
    let mut bytes = Vec::with_capacity(text.len() / 4 * 3);
    for chunk in text.chunks(4) {
        let padding = chunk.iter().rev().take_while(|byte| **byte == b'=').count();
        let mut word = 0_u32;
        for byte in &chunk[..4 - padding] {
            let digit = value(*byte).ok_or_else(|| invalid("a tiktoken token is not base64"))?;
            word = (word << 6) | u32::from(digit);
        }
        word <<= 6 * padding;
        let decoded = word.to_be_bytes();
        bytes.extend_from_slice(&decoded[1..4 - padding]);
    }
    Ok(bytes)
}

fn invalid(message: &str) -> ModelsError {
    ModelsError::InvalidConfig(message.into())
}
