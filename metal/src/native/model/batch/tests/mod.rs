mod capacity;
mod fixture;
mod recovery;

use runtime::backend::SamplingLogits;
use uuid::Uuid;

use super::{DecodeExecution, DecodeInput, LoadedModel, NativeOutput};
use crate::native::{error::Result, step};

#[test]
fn mixed_sampling_preserves_tokens_and_rejoins_packed_decode() -> Result<()> {
    let (mut model, directory) = fixture::load()?;
    let mut rows = (0..5).map(|row| pair(&mut model, row)).collect::<Result<Vec<_>>>()?;
    let mut packed_steps = 0;
    for iteration in 0..128 {
        if iteration == 32 {
            rows.rotate_left(2);
        }
        if iteration == 64 {
            let (input, reference) = rows.remove(1);
            model.release_session(input.session)?;
            model.release_session(reference)?;
            assert!(!model.sessions.contains_key(&input.session));
            rows.push(pair(&mut model, 17)?);
        }
        for (row, (input, _)) in rows.iter_mut().enumerate() {
            input.sampling = policy(row, iteration);
        }
        let inputs = rows.iter().map(|(input, _)| *input).collect::<Vec<_>>();
        let eligible = inputs.iter().filter(|input| model.can_decode_packed_row(input)).count();
        let actual = model.decode_grouped(&inputs)?;
        assert_eq!(actual.len(), rows.len());
        for (row, ((input, reference), (output, execution))) in
            rows.iter_mut().zip(actual).enumerate()
        {
            if let DecodeExecution::Packed { rows } = execution {
                assert_eq!(rows, eligible);
                assert!(rows >= 2);
                packed_steps += 1;
            }
            let expected = model.decode(*reference, input.token, SamplingLogits::Full)?;
            let expected = choose(&model, expected, input.sampling)?;
            let actual = choose(&model, output, input.sampling)?;
            assert_eq!(actual, expected, "row {row}, step {iteration}");
            assert_eq!(
                model.sessions[&input.session].pending.is_some(),
                step::supports_device_token(input.sampling),
                "pipeline must resume after a full-logit step"
            );
            assert_eq!(model.sessions[&input.session].position, model.sessions[reference].position);
            input.token = actual;
        }
    }
    assert!(packed_steps > 128, "mixed policies must retain a packed cohort");
    for (input, reference) in rows {
        model.release_session(input.session)?;
        model.release_session(reference)?;
    }
    assert!(model.sessions.is_empty());
    drop(model);
    std::fs::remove_dir_all(directory)?;
    Ok(())
}

#[test]
fn invalid_group_does_not_consume_any_pending_tokens() -> Result<()> {
    let (mut model, directory) = fixture::load()?;
    let (first, reference_first) = pair(&mut model, 0)?;
    let (second, reference_second) = pair(&mut model, 1)?;
    let before = model.sessions[&first.session].position;
    let bad = DecodeInput {
        token: second.token.wrapping_add(1),
        ..second
    };
    assert!(model.decode_grouped(&[first, bad]).is_err());
    assert_eq!(model.sessions[&first.session].position, before);
    assert_eq!(
        model.sessions[&first.session].pending.as_ref().map(|pending| pending.token_id),
        Some(first.token)
    );
    assert_eq!(
        model.sessions[&second.session].pending.as_ref().map(|pending| pending.token_id),
        Some(second.token)
    );
    assert!(model.decode_grouped(&[first, first]).is_err());
    assert!(model.decode(second.session, bad.token, SamplingLogits::None).is_err());
    assert_eq!(
        model.sessions[&second.session].pending.as_ref().map(|pending| pending.token_id),
        Some(second.token)
    );
    let outputs = model.decode_grouped(&[first, second])?;
    assert_eq!(outputs.len(), 2);
    for session in [first.session, second.session, reference_first, reference_second] {
        model.release_session(session)?;
    }
    drop(model);
    std::fs::remove_dir_all(directory)?;
    Ok(())
}

fn pair(model: &mut LoadedModel, seed: u32) -> Result<(DecodeInput, Uuid)> {
    let prompt = (0..6).map(|token| (seed + token) % 63 + 1).collect::<Vec<_>>();
    let session = Uuid::new_v4();
    let reference = Uuid::new_v4();
    let output = model.prefill(session, &prompt, &[], SamplingLogits::None, None, &mut |_| {})?;
    let token = choose(model, output.output, SamplingLogits::None)?;
    let output = model.prefill(reference, &prompt, &[], SamplingLogits::Full, None, &mut |_| {})?;
    assert_eq!(token, choose(model, output.output, SamplingLogits::Full)?);
    Ok((
        DecodeInput {
            session,
            token,
            sampling: SamplingLogits::None,
        },
        reference,
    ))
}

fn policy(row: usize, step: usize) -> SamplingLogits {
    match (row + step) % 5 {
        0 => SamplingLogits::Full,
        1 => SamplingLogits::Sample {
            vocab_size: 64,
            temperature: 0.7,
            top_k: 0,
            top_p: 1.0,
            draw: [0.1, 0.3, 0.5, 0.7, 0.9][row],
        },
        _ => SamplingLogits::None,
    }
}

fn choose(model: &LoadedModel, output: NativeOutput, sampling: SamplingLogits) -> Result<u32> {
    match output {
        NativeOutput::Greedy(token) => Ok(token),
        NativeOutput::Logits(logits) => {
            match step::device_token(&logits, sampling, model.stream())? {
                Some(token) => Ok(token.item_u32(model.stream())?),
                None => Ok(logits.argmax_u32(model.stream())?),
            }
        },
    }
}
