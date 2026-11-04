use llguidance::{Matcher, api::TopLevelGrammar};
use runtime::backend::SamplingLogits;
use serde_json::json;

use super::{
    super::{
        OutputConstraint, Target, budget::Allowance, flow::advance, grammar, phase::Phase,
        sampling, validate_content,
    },
    TestResult, factory,
};

/// Content constrained to `{"verdict": ...}`, with token 256 opening a
/// reasoning block that token 257 closes.
fn constraint() -> TestResult<OutputConstraint> {
    let factory = factory(true)?;
    let schema = json!({"type":"object","required":["verdict"],"additionalProperties":false,
        "properties":{"verdict":{"enum":["supported","contradicted"]}}});
    let grammar = grammar::content(&schema)?;
    Ok(OutputConstraint {
        matcher: Matcher::new(factory.create_parser(TopLevelGrammar::from_lark(grammar))),
        phase: Phase::Opening {
            blocks: vec![(256, 257)],
            allowance: Allowance::Unlimited,
        },
        vocab: 258,
        validator: jsonschema::validator_for(&schema)?,
        target: Target::Content,
    })
}

fn allowed(c: &mut Option<OutputConstraint>, id: u32) -> TestResult<bool> {
    let SamplingLogits::Masked { mask, .. } = sampling(c, SamplingLogits::None)? else {
        return Err("content grammar left sampling unmasked".into());
    };
    Ok(mask.words()[id as usize / 32] & (1 << (id % 32)) != 0)
}

#[test]
fn content_may_open_reasoning_then_the_schema_owns_the_answer() -> TestResult {
    let factory = factory(true)?;
    let mut c = Some(constraint()?);
    assert!(allowed(&mut c, 256)?);
    assert!(allowed(&mut c, u32::from(b'{'))?);
    assert!(!allowed(&mut c, u32::from(b'S'))?);
    assert!(!advance(&mut c, 256)?);
    assert_eq!(sampling(&mut c, SamplingLogits::None)?, SamplingLogits::None);
    for id in factory.tok_env().tokenize("Thinking may say anything, even {broken.") {
        assert!(!advance(&mut c, id)?);
    }
    assert!(!advance(&mut c, 257)?);
    assert!(!allowed(&mut c, 256)?);
    let answer = r#"{"verdict": "supported"}"#;
    let tokens = factory.tok_env().tokenize(answer);
    let count = tokens.len();
    for (index, id) in tokens.into_iter().enumerate() {
        assert!(allowed(&mut c, id)?);
        assert_eq!(advance(&mut c, id)?, index + 1 == count);
    }
    validate_content(c.as_mut(), answer)?;
    Ok(())
}

#[test]
fn content_without_reasoning_starts_in_the_grammar() -> TestResult {
    let factory = factory(true)?;
    let mut c = Some(constraint()?);
    let answer = r#"{"verdict": "contradicted"}"#;
    for (index, id) in factory.tok_env().tokenize(answer).into_iter().enumerate() {
        advance(&mut c, id)?;
        if index == 0 {
            assert!(!allowed(&mut c, 256)?);
        }
    }
    validate_content(c.as_mut(), answer)?;
    Ok(())
}

#[test]
fn unfinished_content_is_rejected() -> TestResult {
    let factory = factory(true)?;
    let mut c = Some(constraint()?);
    for id in factory.tok_env().tokenize(r#"{"verdict": "sup"#) {
        advance(&mut c, id)?;
    }
    assert!(validate_content(c.as_mut(), r#"{"verdict": "sup"#).is_err());
    Ok(())
}
