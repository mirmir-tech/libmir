//! Parity of every enabled Laya backend with the upstream fp32 CPU runtime.
//!
//! Needs the `convaiinnovations/laya-multilingual` checkpoint directory in
//! `LIBMIR_LAYA_MODEL`; without it the tests have nothing to compare.
#![cfg(any(feature = "cpu", feature = "metal", feature = "cuda"))]

mod reference;

use std::collections::BTreeMap;

use libmir::decision::{DecisionBackend, DecisionModel, DecisionRow};
use reference::{Case, Failure, Reference};

/// Largest accepted logit difference; fp32 reductions run in a different
/// order than `PyTorch`'s.
const LOGIT_TOLERANCE: f32 = 5e-4;

const BACKENDS: &[DecisionBackend] = &[
    #[cfg(feature = "cpu")]
    DecisionBackend::Cpu,
    #[cfg(feature = "metal")]
    DecisionBackend::Metal,
    #[cfg(feature = "cuda")]
    DecisionBackend::Cuda,
];

/// Reference cases of one state and the rows libmir builds for them.
type StateGroup<'a> = (Vec<&'a Case>, Vec<DecisionRow>);

fn reference() -> Result<Reference, Failure> {
    Ok(serde_json::from_str(include_str!("multilingual-fp32.json"))?)
}

fn rows<'a>(
    model: &DecisionModel,
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
            let rows = model.rows(&state, &questions)?;
            for (case, row) in cases.iter().zip(&rows) {
                assert_eq!(row.tokens, case.tokens, "{}/{}: tokens", case.name, case.question);
                assert_eq!(row.markers, case.markers, "{}/{}: markers", case.name, case.question);
            }
            Ok((cases, rows))
        })
        .collect()
}

#[test]
fn every_backend_reproduces_the_reference_rows_and_logits() -> Result<(), Failure> {
    let Some(root) = std::env::var_os("LIBMIR_LAYA_MODEL") else {
        return Ok(());
    };
    let reference = reference()?;
    for &backend in BACKENDS {
        let model = DecisionModel::load(&root, backend)?;
        let mut largest = 0.0_f32;
        for (cases, rows) in rows(&model, &reference)? {
            for (case, actual) in cases.iter().zip(&model.logits(&rows)?) {
                let label = format!("{backend:?} {}/{}", case.name, case.question);
                assert_eq!(actual.len(), case.logits.len(), "{label}: option count");
                for (actual, expected) in actual.iter().zip(&case.logits) {
                    largest = largest.max((actual - expected).abs());
                }
                assert_eq!(argmax(actual), argmax(&case.logits), "{label}: argmax");
            }
        }
        assert!(largest <= LOGIT_TOLERANCE, "{backend:?}: largest logit difference {largest}");
    }
    Ok(())
}

fn argmax(values: &[f32]) -> Option<usize> {
    values
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index)
}
