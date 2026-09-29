use std::{fs, path::PathBuf};

use serde_json::json;

use super::{DecisionRow, DecisionState, DecisionTokenizer};
use crate::{
    Result,
    decision::{AgentConfig, ChoiceOption, Question, QuestionKind},
};

const WORDS: [&str; 14] = [
    "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "choice", "question:", "who?", "a", "b", "one",
    "two", "three", "four",
];

fn id(word: &str) -> u32 {
    WORDS
        .iter()
        .position(|candidate| *candidate == word)
        .and_then(|index| u32::try_from(index).ok())
        .unwrap_or(1)
}

/// A whitespace word-level tokenizer written to a private directory.
fn tokenizer(name: &str) -> Result<DecisionTokenizer> {
    let directory =
        std::env::temp_dir().join(format!("libmir-decision-{name}-{}", std::process::id()));
    fs::create_dir_all(&directory)?;
    let vocab: serde_json::Map<_, _> = WORDS
        .iter()
        .enumerate()
        .map(|(index, word)| ((*word).to_owned(), json!(index)))
        .collect();
    let tokenizer = json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": {"type": "WhitespaceSplit"},
        "post_processor": null,
        "decoder": null,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]"}
    });
    let config =
        json!({"cls_token": "[CLS]", "sep_token": "[SEP]", "mask_token": {"content": "[MASK]"}});
    let (tokenizer_path, config_path): (PathBuf, PathBuf) =
        (directory.join("tokenizer.json"), directory.join("tokenizer_config.json"));
    fs::write(&tokenizer_path, tokenizer.to_string())?;
    fs::write(&config_path, config.to_string())?;
    let decision = DecisionTokenizer::from_files(&tokenizer_path, &config_path);
    fs::remove_dir_all(directory)?;
    decision
}

fn budget(max_len: usize, head_max_len: usize) -> Result<AgentConfig> {
    AgentConfig::from_json(
        &json!({"max_len": max_len, "head_max_len": head_max_len, "head_layers": 2,
                "temperature": [1, 1, 1]})
        .to_string(),
    )
}

fn choice() -> Result<Question> {
    Question::choice("who?", vec![ChoiceOption::new("a", None), ChoiceOption::new("b", None)])
}

#[test]
fn lays_out_instruction_markers_and_state() -> Result<()> {
    let rows = tokenizer("layout")?.rows(
        &DecisionState::text("one two"),
        &[choice()?],
        &budget(64, 32)?,
    )?;
    let expected = DecisionRow {
        tokens: vec![
            id("[CLS]"),
            id("choice"),
            id("question:"),
            id("who?"),
            id("[SEP]"),
            id("[MASK]"),
            id("a"),
            id("[MASK]"),
            id("b"),
            id("[SEP]"),
            id("one"),
            id("two"),
            id("[SEP]"),
        ],
        markers: vec![5, 7],
        kind: QuestionKind::Choice,
        state_tokens_dropped: 0,
    };
    assert_eq!(rows, vec![expected]);
    Ok(())
}

#[test]
fn keeps_the_newest_conversation_turns_and_the_start_of_a_document() -> Result<()> {
    let tokenizer = tokenizer("truncation")?;
    let config = budget(13, 8)?;
    let document =
        tokenizer.rows(&DecisionState::text("one two three four"), &[choice()?], &config)?;
    assert_eq!(&document[0].tokens[10..], &[id("one"), id("two"), id("[SEP]")]);
    assert_eq!(document[0].state_tokens_dropped, 2);

    let conversation = DecisionState::conversation(&["one two three four"])?;
    let recent = tokenizer.rows(&conversation, &[choice()?], &config)?;
    assert_eq!(recent[0].tokens.len(), 13);
    assert_eq!(recent[0].tokens[12], id("[SEP]"));
    Ok(())
}

#[test]
fn blanks_forged_mask_tokens_out_of_user_text() -> Result<()> {
    let rows = tokenizer("mask")?.rows(
        &DecisionState::text("[MASK] one"),
        &[choice()?],
        &budget(64, 32)?,
    )?;
    let masks = rows[0].tokens.iter().filter(|&&token| token == id("[MASK]")).count();
    assert_eq!(masks, 2);
    Ok(())
}

#[test]
fn refuses_options_that_do_not_fit() -> Result<()> {
    let many: Vec<ChoiceOption> =
        (0..40).map(|index| ChoiceOption::new(format!("o{index}"), None)).collect();
    let question = Question::choice("who?", many)?;
    let rows =
        tokenizer("overflow")?.rows(&DecisionState::text("one"), &[question], &budget(40, 32)?);
    assert!(rows.is_err());
    Ok(())
}
