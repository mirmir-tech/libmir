use super::*;
use crate::native::model::recovery::fail_next_drain;

pub(super) fn survivors(model: &mut LoadedModel) -> Result<(Uuid, Uuid, u32)> {
    let first = request(model, 6);
    let second = request(model, 6);
    let mut tokens = Vec::new();
    for request in [&first, &second] {
        let output = model.prefill(
            request.session_id,
            &request.prompt_tokens,
            &[],
            SamplingLogits::Full,
            None,
            &mut |_| {},
        )?;
        tokens.push(token(model, output.output)?);
    }
    assert_eq!(tokens[0], tokens[1]);
    Ok((first.session_id, second.session_id, tokens[0]))
}

pub(super) fn arm_drain_failure() {
    fail_next_drain();
}

pub(super) fn recover(
    model: &mut LoadedModel,
    retired: usize,
    survivor: (Uuid, Uuid, u32),
) -> Result<()> {
    let states = model.retained_execution_states().ok_or_else(|| {
        Error::InvalidPrefillBatch("failed prefill did not retain its resources".into())
    })?;
    assert_eq!(states.len(), retired);
    assert!(model.decode(survivor.0, survivor.2, SamplingLogits::Full).is_err());
    let request = request(model, 4);
    assert!(
        model
            .prefill(
                request.session_id,
                &request.prompt_tokens,
                &[],
                SamplingLogits::Full,
                None,
                &mut |_| {}
            )
            .is_err()
    );
    model.recover_execution()?;
    Ok(())
}

pub(super) fn continue_and_release(
    model: &mut LoadedModel,
    (first, second, mut next): (Uuid, Uuid, u32),
) -> Result<()> {
    for _ in 0..3 {
        let output = model.decode(first, next, SamplingLogits::Full)?;
        let expected = token(model, output)?;
        let output = model.decode(second, next, SamplingLogits::Full)?;
        next = token(model, output)?;
        assert_eq!(next, expected);
    }
    model.release_session(first)?;
    model.release_session(second)?;
    model.clear_prefix_cache();
    model.flush_decode_graphs()?;
    assert_eq!(model.stream().paged_arenas().resident_arenas()?, 0);
    Ok(())
}

fn token(model: &LoadedModel, output: NativeOutput) -> Result<u32> {
    match output {
        NativeOutput::Greedy(token) => Ok(token),
        NativeOutput::Logits(logits) => Ok(logits.argmax_u32(model.stream())?),
    }
}
