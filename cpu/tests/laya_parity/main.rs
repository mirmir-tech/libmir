//! Parity of the CPU Laya backend with the upstream fp32 CPU runtime.
//!
//! Needs the `convaiinnovations/laya-multilingual` checkpoint directory in
//! `LIBMIR_LAYA_MODEL`; without it the tests have nothing to compare.

mod reference;

use std::{collections::BTreeMap, path::PathBuf};

use libmir_cpu::CpuDecisionModel;
use models::decision::{DecisionCheckpoint, DecisionRow};
use reference::{Case, Failure, Reference};

/// Largest accepted logit difference; fp32 reductions run in a different
/// order than `PyTorch`'s.
const LOGIT_TOLERANCE: f32 = 5e-4;

fn setup() -> Result<Option<(DecisionCheckpoint, Reference)>, Failure> {
    let Some(root) = std::env::var_os("LIBMIR_LAYA_MODEL") else {
        return Ok(None);
    };
    let fixture =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/laya_parity/multilingual-fp32.json");
    let reference: Reference = serde_json::from_str(&std::fs::read_to_string(fixture)?)?;
    Ok(Some((DecisionCheckpoint::inspect(root)?, reference)))
}

/// Reference cases of one state and the rows libmir builds for them.
type StateGroup<'a> = (Vec<&'a Case>, Vec<DecisionRow>);

fn rows<'a>(
    checkpoint: &DecisionCheckpoint,
    reference: &'a Reference,
) -> Result<Vec<StateGroup<'a>>, Failure> {
    let mut groups: BTreeMap<&str, Vec<&Case>> = BTreeMap::new();
    for case in &reference.cases {
        groups.entry(case.name.as_str()).or_default().push(case);
    }
    groups
        .into_values()
        .map(|cases| {
            let state = cases[0].state()?;
            assert_eq!(
                state.as_str(),
                cases[0].serialized_state,
                "{}: serialized state",
                cases[0].name
            );
            let questions =
                cases.iter().map(|case| case.question()).collect::<Result<Vec<_>, _>>()?;
            let rows = checkpoint.tokenizer.rows(&state, &questions, &checkpoint.agent)?;
            Ok((cases, rows))
        })
        .collect()
}

#[test]
fn builds_the_reference_token_rows() -> Result<(), Failure> {
    let Some((checkpoint, reference)) = setup()? else {
        return Ok(());
    };
    for (cases, rows) in rows(&checkpoint, &reference)? {
        for (case, row) in cases.iter().zip(&rows) {
            assert_eq!(row.tokens, case.tokens, "{}/{}: tokens", case.name, case.question);
            assert_eq!(row.markers, case.markers, "{}/{}: markers", case.name, case.question);
        }
    }
    Ok(())
}

#[test]
fn reproduces_the_reference_logits() -> Result<(), Failure> {
    let Some((checkpoint, reference)) = setup()? else {
        return Ok(());
    };
    let model = CpuDecisionModel::load(&checkpoint)?;
    let mut largest = 0.0_f32;
    for (cases, rows) in rows(&checkpoint, &reference)? {
        let logits = model.logits(&rows)?;
        for (case, actual) in cases.iter().zip(&logits) {
            let label = format!("{}/{}", case.name, case.question);
            assert_eq!(actual.len(), case.logits.len(), "{label}: option count");
            for (actual, expected) in actual.iter().zip(&case.logits) {
                let difference = (actual - expected).abs();
                largest = largest.max(difference);
                assert!(difference <= LOGIT_TOLERANCE, "{label}: {actual} vs {expected}");
            }
            assert_eq!(argmax(actual), argmax(&case.logits), "{label}: argmax");
        }
    }
    assert!(largest <= LOGIT_TOLERANCE, "largest logit difference {largest}");
    Ok(())
}

fn argmax(values: &[f32]) -> Option<usize> {
    values
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index)
}
