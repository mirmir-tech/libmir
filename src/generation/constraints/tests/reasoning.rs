use llguidance::{Matcher, api::TopLevelGrammar};
use models::chat::ToolCallPrefix;
use runtime::backend::SamplingLogits;
use serde_json::json;

use super::{
    super::{ToolConstraint, advance, grammar, phase::Phase, sampling, validate},
    TestResult, factory,
};

fn constraint() -> TestResult<ToolConstraint> {
    let factory = factory(true)?;
    let schema = json!({"type":"object","required":["verdict"],"additionalProperties":false,
        "properties":{"verdict":{"enum":["supported","contradicted"]}}});
    let grammar = grammar::with_prefix(
        &grammar::compile(&schema, None)?,
        &ToolCallPrefix::XmlFunction("verify".into()),
    )?;
    Ok(ToolConstraint {
        matcher: Matcher::new(factory.create_parser(TopLevelGrammar::from_lark(grammar))),
        phase: Phase::Reasoning { end: 258 },
        vocab: 258,
        validator: jsonschema::validator_for(&schema)?,
    })
}

#[test]
fn reasoning_transitions_once_then_schema_owns_envelope_arguments_and_stop() -> TestResult {
    let factory = factory(true)?;
    let mut c = Some(constraint()?);
    assert_eq!(sampling(&mut c, SamplingLogits::None)?, SamplingLogits::None);
    for id in factory
        .tok_env()
        .tokenize("Reasoning can mention <tool_call> without opening a call.")
    {
        assert!(!advance(&mut c, id)?);
    }
    assert!(!advance(&mut c, 258)?);
    assert!(c.as_ref().ok_or("missing constraint")?.is_tool());
    assert!(matches!(sampling(&mut c, SamplingLogits::None)?, SamplingLogits::Masked { .. }));
    let text = "<tool_call>\n<function=verify>\n<parameter=verdict>\"supported\"</parameter>\n</function>\n</tool_call>";
    let tokens = factory.tok_env().tokenize(text);
    let count = tokens.len();
    for (index, id) in tokens.into_iter().enumerate() {
        assert!(c.as_mut().ok_or("missing constraint")?.matcher.compute_mask()?.is_allowed(id));
        assert_eq!(advance(&mut c, id)?, index + 1 == count);
    }
    validate(
        c.as_mut(),
        r#"[{"id":"call","type":"function","function":{"name":"verify","arguments":{"verdict":"supported"}}}]"#,
    )?;
    Ok(())
}

#[test]
fn final_channel_cannot_emit_prose_wrong_tool_or_invalid_verdict() -> TestResult {
    let factory = factory(true)?;
    for text in [
        "Sure",
        "<tool_call>\n<function=other>\n",
        "<tool_call>\n<function=verify>\n<parameter=verdict>\"maybe\"",
    ] {
        let mut c = constraint()?;
        c.consume(258)?;
        let mut rejected = false;
        for id in factory.tok_env().tokenize(text) {
            if !c.matcher.compute_mask()?.is_allowed(id) {
                rejected = true;
                break;
            }
            c.consume(id)?;
        }
        assert!(rejected, "accepted {text}");
    }
    Ok(())
}

#[test]
fn truncated_reasoning_or_tool_never_becomes_a_valid_empty_call() -> TestResult {
    let mut c = constraint()?;
    assert!(!c.complete()?);
    assert!(validate(Some(&mut c), "").is_err());
    c.consume(258)?;
    assert!(!c.complete()?);
    assert!(!c.matcher.compute_mask()?.is_allowed(256));
    assert!(validate(Some(&mut c), "[]").is_err());
    Ok(())
}
