use llguidance::{Matcher, api::TopLevelGrammar};
use serde_json::json;

use super::{
    super::{OutputConstraint, Target, flow::advance, grammar, phase::Phase, validate_content},
    TestResult, factory,
};

#[test]
fn numeric_roots_continue_until_eos_instead_of_stopping_at_a_digit() -> TestResult {
    for (schema, output) in [
        (json!({"type":"integer"}), "123"),
        (json!({"type":"number"}), "-12.34e+5"),
        (json!({"enum":[1, 123]}), "123"),
        (json!({"anyOf":[{"type":"integer"},{"type":"object"}]}), "123"),
    ] {
        let factory = factory(true)?;
        let mut constraint = Some(OutputConstraint {
            matcher: Matcher::new(
                factory.create_parser(TopLevelGrammar::from_lark(grammar::content(&schema)?)),
            ),
            phase: Phase::Grammar,
            vocab: 258,
            validator: jsonschema::validator_for(&schema)?,
            target: Target::Content,
        });
        let tokens = factory.tok_env().tokenize(output);
        for (index, token) in tokens.iter().enumerate() {
            let completed = advance(&mut constraint, *token)?;
            if index + 1 < tokens.len() {
                assert!(!completed, "premature end of {output} at {index}");
            }
        }
        let active = constraint.as_mut().ok_or("missing constraint")?;
        if !active.matcher.is_stopped() {
            assert!(active.matcher.compute_mask()?.is_allowed(256));
            assert!(advance(&mut constraint, 256)?);
        }
        validate_content(constraint.as_mut(), output)?;
    }
    Ok(())
}
