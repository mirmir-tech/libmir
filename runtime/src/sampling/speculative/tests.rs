use super::*;
use crate::backend::SamplingLogits;

const SAMPLING: DraftSampling = DraftSampling { temperature: 1.0, top_p: 1.0, top_k: 0 };

/// Frequencies of the first emitted token over many seeds.
fn first_tokens(rows: &[&[f32]], drafts: &[u32], sampling: DraftSampling) -> Result<[f64; 3]> {
    let mut counts = [0.0; 3];
    let trials = 40_000_u64;
    for seed in 0..trials {
        let (kept, token) = verify_drafts(rows, drafts, sampling, seed)?;
        let first = if kept > 0 {
            drafts[0]
        } else {
            token
        };
        counts[first as usize] += 1.0;
    }
    Ok(counts.map(|count| count / f64::from(u32::try_from(trials).unwrap_or(u32::MAX))))
}

#[test]
fn first_token_keeps_the_target_distribution() -> Result<()> {
    // p = softmax([0, ln 2, ln 3]) = [1/6, 2/6, 3/6]; the draft is token 1.
    let logits = [0.0, 2.0_f32.ln(), 3.0_f32.ln()];
    let frequencies = first_tokens(&[&logits, &logits], &[1], SAMPLING)?;
    for (frequency, expected) in frequencies.iter().zip([1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0]) {
        assert!((frequency - expected).abs() < 0.01, "{frequencies:?}");
    }
    Ok(())
}

#[test]
fn a_draft_outside_top_p_is_always_rejected() -> Result<()> {
    let logits = [0.0, 0.0, 10.0];
    let sampling = DraftSampling { top_p: 0.5, ..SAMPLING };
    for seed in 0..100 {
        assert_eq!(verify_drafts(&[&logits, &logits], &[0], sampling, seed)?, (0, 2));
    }
    Ok(())
}

#[test]
fn certain_drafts_are_all_kept() -> Result<()> {
    let certain = [-100.0, 100.0, -100.0];
    let rows: [&[f32]; 3] = [&certain, &certain, &[0.0, 0.0, 100.0]];
    assert_eq!(verify_drafts(&rows, &[1, 1], SAMPLING, 7)?, (2, 2));
    Ok(())
}

#[test]
fn sampled_steps_are_the_device_sampled_kinds() {
    let step = SampledStep::of(&SamplingLogits::SampleTopK {
        k: 20,
        vocab_size: 100,
        temperature: 0.7,
        draw: 0.25,
    });
    assert_eq!(
        step,
        Some(SampledStep {
            sampling: DraftSampling { temperature: 0.7, top_p: 1.0, top_k: 20 },
            draw: 0.25,
            vocab_size: 100,
        })
    );
    assert_eq!(SampledStep::of(&SamplingLogits::None), None);
    assert_eq!(SampledStep::of(&SamplingLogits::Full), None);
}

#[test]
fn seeds_put_the_position_above_the_draw() -> Result<()> {
    assert_eq!(speculative_seed(0.5, 3)?, u64::from(0.5_f32.to_bits()) | (3 << 32));
    Ok(())
}
