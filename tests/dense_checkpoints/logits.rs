use super::{
    TestResult,
    fixture::{LogitsReference, Reference, active_target, require, validation_error},
};

pub fn validate_logits(output: &libmir::PrefillOutput, reference: &Reference) -> TestResult<u32> {
    let expected = reference.logits(&active_target());
    let actual = output
        .logits
        .as_ref()
        .ok_or_else(|| validation_error("prefill did not return full logits"))?;
    require(
        actual.values.len() == reference.vocab_size,
        "first-token logits width differs from checkpoint vocabulary",
    )?;
    require(
        actual.values.iter().all(|value| value.is_finite()),
        "first-token logits contain a non-finite value",
    )?;
    let mut top = actual.values.iter().copied().enumerate().collect::<Vec<_>>();
    top.sort_unstable_by(|left, right| right.1.total_cmp(&left.1));
    top.truncate(expected.token_ids.len());
    if expected.normalized {
        let maximum = actual.values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let partition =
            maximum + actual.values.iter().map(|value| (*value - maximum).exp()).sum::<f32>().ln();
        for (_token, score) in &mut top {
            *score -= partition;
        }
    }
    let token_ids = top
        .iter()
        .map(|(token, _)| u32::try_from(*token))
        .collect::<Result<Vec<_>, _>>()?;
    validate_top_ids(&token_ids, &top, expected)?;
    validate_top_scores(&token_ids, &top, expected)?;
    token_ids
        .first()
        .copied()
        .ok_or_else(|| validation_error("first-token top-k is empty"))
}

fn validate_top_ids(
    token_ids: &[u32],
    top: &[(usize, f32)],
    reference: &LogitsReference,
) -> TestResult<()> {
    require(top.len() == token_ids.len(), "top-k token and score widths differ")?;
    let expected_ids = &reference.token_ids;
    let actual_only = top
        .iter()
        .filter(|(token, _)| !expected_ids.contains(&u32::try_from(*token).unwrap_or(u32::MAX)))
        .map(|(_, score)| *score)
        .collect::<Vec<_>>();
    let missing = expected_ids
        .iter()
        .enumerate()
        .filter(|(_, token)| !token_ids.contains(token))
        .map(|(rank, _)| reference.scores[rank])
        .collect::<Vec<_>>();
    require(
        tied_scores(&actual_only, &missing, reference.absolute_tolerance),
        format!(
            "first-token top-k ID set differs outside a boundary tie: actual={top:?}, expected={:?}",
            expected_ids.iter().zip(&reference.scores).collect::<Vec<_>>()
        ),
    )?;
    for (rank, token) in token_ids.iter().enumerate() {
        if let Some(expected_rank) = expected_ids.iter().position(|expected| expected == token) {
            require(
                rank == expected_rank
                    || tied(
                        reference.scores[rank],
                        reference.scores[expected_rank],
                        reference.absolute_tolerance,
                    ),
                format!("top-k token {token} crossed a non-tied reference rank"),
            )?;
        }
    }
    Ok(())
}

fn validate_top_scores(
    token_ids: &[u32],
    top: &[(usize, f32)],
    reference: &LogitsReference,
) -> TestResult<()> {
    for (rank, (token, (_, actual))) in token_ids.iter().zip(top).enumerate() {
        let expected = reference
            .token_ids
            .iter()
            .position(|candidate| candidate == token)
            .map_or(reference.scores[rank], |index| reference.scores[index]);
        require(
            tied(*actual, expected, reference.absolute_tolerance),
            format!(
                "first-token top-k score exceeds reference tolerance: rank={rank}, token={token}, \
                 actual={actual}, expected={expected}, tolerance={}, top={top:?}",
                reference.absolute_tolerance,
            ),
        )?;
    }
    Ok(())
}

fn tied_scores(actual: &[f32], expected: &[f32], tolerance: f32) -> bool {
    if actual.len() != expected.len() {
        return false;
    }
    let mut actual = actual.to_vec();
    let mut expected = expected.to_vec();
    actual.sort_unstable_by(f32::total_cmp);
    expected.sort_unstable_by(f32::total_cmp);
    actual
        .iter()
        .zip(expected)
        .all(|(actual, expected)| tied(*actual, expected, tolerance))
}

fn tied(actual: f32, expected: f32, tolerance: f32) -> bool {
    (actual - expected).abs() <= tolerance
}
