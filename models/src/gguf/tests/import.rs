//! Real-checkpoint gates for the GGUF import: run with
//! `LIBMIR_GGUF=/path/model.gguf LIBMIR_GGUF_SOURCE=/path/safetensors-dir`
//! `cargo test -p libmir-models -- --ignored gguf::tests::import`.

use std::path::PathBuf;

use tokenizers::Tokenizer;

use super::{GgufModel, Import, TestResult};

/// Text that exercises the split pattern, NFC, numbers, code, whitespace runs
/// and every special-token family of a chat prompt.
const SAMPLES: [&str; 8] = [
    "Hello, world! It's a test: we'll see 12345 numbers and 3.14159.",
    "Zażółć gęślą jaźń — café, naïve, Ａｂｃ, e\u{301}, 日本語のテキスト, 🚀🔥",
    "fn main() {\n    println!(\"{}\", 1 + 2);\n}\n\n\n\t\ttabs  and   spaces   ",
    "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Kraków\"}}\n</tool_call>",
    "   leading spaces and trailing   ",
    "I'M SHOUTING, YOU'RE NOT, WE'VE BEEN, THEY'LL GO, HE'D SAY",
    "",
];

fn paths() -> Result<(PathBuf, PathBuf), Box<dyn std::error::Error + Send + Sync>> {
    Ok((
        std::env::var("LIBMIR_GGUF")?.into(),
        std::env::var("LIBMIR_GGUF_SOURCE")?.into(),
    ))
}

#[test]
#[ignore = "needs a real GGUF file (LIBMIR_GGUF) and its source tokenizer (LIBMIR_GGUF_SOURCE)"]
fn import_tokenizer_matches_the_source_tokenizer() -> TestResult {
    let (gguf, source) = paths()?;
    let import = Import::derive(&GgufModel::open(&gguf)?)?;
    let derived = Tokenizer::from_bytes(serde_json::to_vec(&import.tokenizer)?)?;
    let original = Tokenizer::from_file(source.join("tokenizer.json"))?;
    for sample in SAMPLES {
        let left = derived.encode(sample, false)?;
        let right = original.encode(sample, false)?;
        assert_eq!(left.get_ids(), right.get_ids(), "{sample:?}");
        assert_eq!(
            derived.decode(left.get_ids(), false)?,
            original.decode(right.get_ids(), false)?
        );
    }
    Ok(())
}

#[test]
#[ignore = "needs a real GGUF file (LIBMIR_GGUF) and its source checkpoint (LIBMIR_GGUF_SOURCE)"]
fn import_config_and_template_match_the_source() -> TestResult {
    let (gguf, source) = paths()?;
    let model = GgufModel::open(&gguf)?;
    let import = Import::derive(&model)?;
    let original: serde_json::Value =
        serde_json::from_slice(&std::fs::read(source.join("config.json"))?)?;
    for key in [
        "hidden_size",
        "num_hidden_layers",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "rms_norm_eps",
        "rope_theta",
        "max_position_embeddings",
        "tie_word_embeddings",
    ] {
        let number =
            |value: &serde_json::Value| value.as_f64().or_else(|| value.as_bool().map(f64::from));
        // Only fields the source declares (latent attention has no head_dim).
        let Some(expected) = number(&original[key]) else {
            continue;
        };
        let Some(derived) = number(&import.config[key]) else {
            return Err(format!("{key} is missing").into());
        };
        assert!(
            (derived - expected).abs() <= expected.abs() * 1e-9,
            "{key}: {derived} != {expected}"
        );
    }
    // The template comes verbatim from the GGUF, which may embed an older
    // revision than the source repository (Qwen/Qwen3-8B-GGUF does).
    assert_eq!(
        import.tokenizer_config["chat_template"].as_str(),
        model.header().string("tokenizer.chat_template")
    );
    Ok(())
}
