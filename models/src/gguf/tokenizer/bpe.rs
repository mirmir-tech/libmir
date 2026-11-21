use serde_json::{Value, json};

use super::{GgufHeader, Vocabulary, strings, vocabulary};
use crate::error::{ModelsError, Result};

/// A byte-level BPE pre-tokenizer as the source model's `tokenizer.json`
/// declares it.
struct PreTokenizer {
    /// Splits applied in order, each isolating its matches.
    patterns: &'static [&'static str],
    /// Then splits every digit off (`Digits` with `individual_digits`).
    individual_digits: bool,
    nfc: bool,
    ignore_merges: bool,
}

/// `tokenizer.ggml.pre` → the source tokenizer. `qwen2`: Qwen2/Qwen3
/// `tokenizer.json` (Qwen/Qwen3-8B revision b968826), which llama.cpp v1.0
/// `llama-vocab.cpp` also attributes to `qwen2`.
fn pre_tokenizer(name: &str) -> Option<PreTokenizer> {
    match name {
        "qwen2" => Some(PreTokenizer {
            patterns: &[
                r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            ],
            individual_digits: false,
            nfc: true,
            ignore_merges: false,
        }),
        // o200k: openai/gpt-oss-20b `tokenizer.json` (revision 6cee5e8), the
        // pattern llama.cpp v1.0.0 lists for `LLAMA_VOCAB_PRE_TYPE_GPT4O`.
        "gpt-4o" => Some(PreTokenizer {
            patterns: &[
                r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            ],
            individual_digits: false,
            nfc: false,
            ignore_merges: true,
        }),
        // Llama 3: unsloth/Llama-3.1-8B-Instruct `tokenizer.json` (revision
        // 4699cc7), llama.cpp v1.0.0 `LLAMA_VOCAB_PRE_TYPE_LLAMA3`.
        "llama-bpe" => Some(PreTokenizer {
            patterns: &[
                r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            ],
            individual_digits: false,
            nfc: false,
            ignore_merges: true,
        }),
        // Script-class splits: newlines, cased-script letter runs, ASCII and
        // full-width punctuation, trailing spaces, CJK runs, then single
        // digits. Escaped so that no Unicode normalization reorders the
        // ranges. Source: llama.cpp `LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_LLM`,
        // deepseek-ai/DeepSeek-V2-Lite-Chat `tokenizer.json`.
        "deepseek-llm" => Some(PreTokenizer {
            patterns: &[
                "[\r\n]",
                "\\s?[A-Za-z\u{B5}\u{C0}-\u{D6}\u{D8}-\u{F6}\u{F8}-\u{1BA}\u{1BC}-\u{1BF}\u{1C4}-\u{293}\u{295}-\u{2AF}\u{370}-\u{373}\u{376}\u{377}\u{37B}-\u{37D}\u{37F}\u{386}\u{388}-\u{38A}\u{38C}\u{38E}-\u{3A1}\u{3A3}-\u{3F5}\u{3F7}-\u{481}\u{48A}-\u{52F}\u{531}-\u{556}\u{10A0}-\u{10C5}\u{13A0}-\u{13F5}\u{13F8}-\u{13FD}\u{1C90}-\u{1CBA}\u{1CBD}-\u{1CBF}\u{1D00}-\u{1D2B}\u{1D6B}-\u{1D77}\u{1D79}-\u{1D9A}\u{1E00}-\u{1F15}\u{1F18}-\u{1F1D}\u{1F20}-\u{1F45}\u{1F48}-\u{1F4D}\u{1F50}-\u{1F57}\u{1F59}\u{1F5B}\u{1F5D}\u{1F5F}-\u{1F7D}\u{1F80}-\u{1FB4}\u{1FB6}-\u{1FBC}\u{1FBE}\u{1FC2}-\u{1FC4}\u{1FC6}-\u{1FCC}\u{1FD0}-\u{1FD3}\u{1FD6}-\u{1FDB}\u{1FE0}-\u{1FEC}\u{1FF2}-\u{1FF4}\u{1FF6}-\u{1FFC}\u{2102}\u{2107}\u{210A}-\u{2113}\u{2115}\u{2119}-\u{211D}\u{2124}\u{2126}\u{2128}\u{212A}-\u{212D}\u{212F}-\u{2134}\u{2139}\u{213C}-\u{213F}\u{2145}-\u{2149}\u{214E}\u{2183}\u{2184}\u{2C00}-\u{2C7B}\u{2C7E}-\u{2CE4}\u{2CEB}-\u{2CEE}\u{2CF2}\u{2CF3}\u{A640}-\u{A66D}\u{A680}-\u{A69B}\u{A722}-\u{A76F}\u{A771}-\u{A787}\u{A78B}-\u{A78E}\u{AB70}-\u{ABBF}\u{FB00}-\u{FB06}\u{FB13}-\u{FB17}\u{FF21}-\u{FF3A}\u{FF41}-\u{FF5A}\u{10400}-\u{1044F}\u{104B0}-\u{104D3}\u{104D8}-\u{104FB}\u{10C80}-\u{10CB2}\u{10CC0}-\u{10CF2}\u{118A0}-\u{118DF}\u{1E900}-\u{1E943}]+",
                "\\s?[!-/:-~\u{FF01}-\u{FF0F}\u{FF1A}-\u{FF5E}\u{2018}-\u{201F}\u{3000}-\u{3002}]+",
                "\\s+$",
                "[\u{4E00}-\u{9FA5}\u{800}-\u{4E00}\u{AC00}-\u{D7FF}]+",
            ],
            individual_digits: true,
            nfc: false,
            ignore_merges: false,
        }),
        // Tekken: the o200k word classes without contractions, single
        // digits. Source: mistralai/Mistral-Small-4-119B-2603 `tokenizer.json`
        // (a11f36b), which llama.cpp lists as `tekken` and `pixtral`.
        "tekken" | "pixtral" => Some(PreTokenizer {
            patterns: &[
                r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            ],
            individual_digits: false,
            nfc: false,
            ignore_merges: true,
        }),
        // The tiktoken Han-run split; tiktoken keeps whole matched tokens.
        "kimi-k2" => Some(PreTokenizer {
            patterns: &[crate::tokenizer::HAN_RUNS_CASED_WORDS],
            individual_digits: false,
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
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": pre
            .patterns
            .iter()
            .map(|pattern| json!({"type": "Split", "pattern": {"Regex": pattern}, "behavior": "Isolated", "invert": false}))
            .chain(pre.individual_digits.then(|| json!({"type": "Digits", "individual_digits": true})))
            .chain([byte_level.clone()])
            .collect::<Vec<_>>()},
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
