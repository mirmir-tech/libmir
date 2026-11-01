use llguidance::{Matcher, api::TopLevelGrammar};
use serde_json::json;

use super::{TestResult, factory, grammar};
use crate::generation::constraints::drafts::masks;

const VOCAB: usize = 257;
const TEXT: &str = "<parameter=count>12</parameter>\n</function>\n</tool_call>";

fn matcher() -> TestResult<(Matcher, Vec<u32>)> {
    let factory = factory(false)?;
    let schema =
        json!({"type":"object","properties":{"count":{"type":"integer"}},"required":["count"]});
    let grammar = grammar::compile(&schema, None)?;
    let matcher = Matcher::new(factory.create_parser(TopLevelGrammar::from_lark(grammar)));
    Ok((matcher, factory.tok_env().tokenize(TEXT)))
}

fn allowed(mask: &runtime::backend::TokenMask, token: u32) -> bool {
    let token = token as usize;
    mask.words()[token / 32] >> (token % 32) & 1 == 1
}

#[test]
fn masked_drafts_carry_the_mask_after_each_draft() -> TestResult {
    let (mut matcher, tokens) = matcher()?;
    let before = matcher.compute_mask()?;
    let masked = masks(&mut matcher, &tokens[..4], VOCAB)?;
    assert_eq!(masked.iter().map(|draft| draft.token).collect::<Vec<_>>(), tokens[..4]);
    for (draft, following) in masked.iter().zip(&tokens[1..]) {
        assert!(allowed(&draft.mask, *following));
    }
    assert_eq!(matcher.compute_mask()?, before, "the matcher moved");
    Ok(())
}

#[test]
fn masking_stops_at_the_first_draft_the_grammar_rejects() -> TestResult {
    let (mut matcher, tokens) = matcher()?;
    let drafts = [tokens[0], tokens[1], u32::from(b'#'), tokens[3]];
    assert_eq!(masks(&mut matcher, &drafts, VOCAB)?.len(), 2);
    for token in &tokens {
        matcher.consume_token(*token)?;
    }
    assert!(matcher.is_accepting()?);
    Ok(())
}

#[test]
fn a_draft_that_ends_the_grammar_is_left_out() -> TestResult {
    let (mut matcher, tokens) = matcher()?;
    let (head, tail) = tokens.split_at(tokens.len() - 3);
    for token in head {
        matcher.consume_token(*token)?;
    }
    let masked = masks(&mut matcher, tail, VOCAB)?;
    assert!(masked.len() < tail.len());
    assert_eq!(masked.iter().map(|draft| draft.token).collect::<Vec<_>>(), tail[..masked.len()]);
    for token in tail {
        matcher.consume_token(*token)?;
    }
    assert!(matcher.is_accepting()?);
    Ok(())
}
