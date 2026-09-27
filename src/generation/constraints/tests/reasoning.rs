use llguidance::{Matcher, api::TopLevelGrammar};
use models::chat::ToolCallPrefix;
use runtime::backend::SamplingLogits;
use serde_json::json;

use super::{
    super::{ToolConstraint, advance, grammar, phase::Phase, sampling},
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
        phase: Phase::Reasoning {
            end: 258,
            allowance: super::super::budget::Allowance::Unlimited,
        },
        vocab: 258,
        validator: jsonschema::validator_for(&schema)?,
    })
}

fn validate(constraint: Option<&mut ToolConstraint>, calls: &str) -> crate::Result<String> {
    use foundation::conversation::{Conversation, FunctionDefinition, Tool, ToolChoice};
    let conversation = Conversation {
        tools: vec![Tool {
            kind: "function".into(),
            function: FunctionDefinition {
                name: "verify".into(),
                description: None,
                parameters: json!({"type":"object","properties":{"verdict":{"type":"string"}}}),
            },
        }],
        tool_choice: ToolChoice::Function("verify".into()),
        ..Conversation::default()
    };
    super::super::normalize(constraint, calls, &conversation)
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
    let error =
        validate(Some(&mut c), "<tool_call><function=verify><parameter=verdict>\"unfinished")
            .err()
            .ok_or("truncated tool accepted")?;
    assert!(error.to_string().contains("ended before completion"));
    // Without a grammar this remains a parser failure, never silent success.
    let error = validate(None, "<tool_call><function=verify><parameter=verdict>\"unfinished")
        .err()
        .ok_or("malformed tool accepted")?;
    assert!(error.to_string().contains("unterminated parameter"));
    Ok(())
}

#[test]
fn budget_masks_only_the_delimiter_then_releases_to_the_tool_grammar() -> TestResult {
    use std::num::NonZeroUsize;

    use super::super::budget::Allowance;
    let mut constraint = constraint()?;
    constraint.phase = Phase::Reasoning {
        end: 255,
        allowance: Allowance::new(NonZeroUsize::new(2)),
    };
    let mut c = Some(constraint);
    assert!(!advance(&mut c, 65)?);
    assert_eq!(sampling(&mut c, SamplingLogits::None)?, SamplingLogits::None);
    let settings = models::generation::GenerationSettings {
        max_tokens: 512,
        min_tokens: 0,
        ignore_eos: false,
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
    };
    let mut cycle =
        crate::generation::CycleRecovery::new(settings, Some(7), 258, &SamplingLogits::None, None)?;
    let repeated = [7, 11, 13, 17].repeat(48);
    cycle.observe(&[], &repeated);
    assert_eq!(cycle.sampling(SamplingLogits::None), SamplingLogits::Full);
    assert!(!super::super::advance_generation(&mut c, 66, &mut cycle)?);
    assert_eq!(cycle.sampling(SamplingLogits::None), SamplingLogits::None);
    for policy in [SamplingLogits::None, SamplingLogits::Full] {
        let SamplingLogits::Masked { mask, sampling: policy } = sampling(&mut c, policy)? else {
            return Err("budget failed to mask the delimiter".into());
        };
        assert_eq!(policy, runtime::backend::DeviceSampling::Greedy);
        assert_eq!(mask.words().iter().map(|word| word.count_ones()).sum::<u32>(), 1);
        assert_eq!(mask.words()[255 / 32], 1 << (255 % 32));
    }
    assert!(!advance(&mut c, 255)?);
    assert!(c.as_ref().ok_or("missing constraint")?.is_tool());
    assert!(c.as_ref().ok_or("missing constraint")?.phase.budget_exit().is_none());
    assert!(matches!(sampling(&mut c, SamplingLogits::None)?, SamplingLogits::Masked { .. }));
    assert!(validate(c.as_mut(), "[]").is_err());
    Ok(())
}

#[test]
fn natural_reasoning_end_cancels_unused_allowance() -> TestResult {
    let mut c = constraint()?;
    c.phase = Phase::Reasoning {
        end: 258,
        allowance: super::super::budget::Allowance::new(std::num::NonZeroUsize::new(10)),
    };
    c.consume(65)?;
    c.consume(258)?;
    assert!(c.is_tool());
    assert!(c.phase.budget_exit().is_none());
    Ok(())
}

#[test]
fn native_stop_closes_reasoning_without_accepting_its_draft_tool() -> TestResult {
    use super::super::transition_stop;

    let factory = factory(true)?;
    let mut c = Some(constraint()?);
    let call = "<tool_call>\n<function=verify>\n<parameter=verdict>\"supported\"</parameter>\n</function>\n</tool_call>";
    // The real reproducer wrote a complete draft in reasoning, then EOS,
    // before its allowance. A draft is neither a real call nor an approval.
    for token in factory.tok_env().tokenize(call) {
        assert_eq!(transition_stop(c.as_ref(), token, &[256]), token);
        assert!(!advance(&mut c, token)?);
    }
    assert!(!c.as_ref().ok_or("constraint")?.is_tool());
    assert!(validate(c.as_mut(), "").is_err());
    let delimiter = transition_stop(c.as_ref(), 256, &[256]);
    assert_eq!(delimiter, 258);
    assert!(!advance(&mut c, delimiter)?);
    assert!(c.as_ref().ok_or("constraint")?.is_tool());
    assert!(validate(c.as_mut(), "").is_err());
    assert_eq!(transition_stop(c.as_ref(), 256, &[256]), 256);
    assert_eq!(transition_stop(None, 256, &[256]), 256);
    assert!(!c.as_mut().ok_or("constraint")?.matcher.compute_mask()?.is_allowed(256));
    let tokens = factory.tok_env().tokenize(call);
    let count = tokens.len();
    for (index, token) in tokens.into_iter().enumerate() {
        assert!(c.as_mut().ok_or("constraint")?.matcher.compute_mask()?.is_allowed(token));
        assert_eq!(advance(&mut c, token)?, index + 1 == count);
    }
    validate(
        c.as_mut(),
        r#"[{"id":"call","type":"function","function":{"name":"verify","arguments":{"verdict":"supported"}}}]"#,
    )?;
    Ok(())
}
