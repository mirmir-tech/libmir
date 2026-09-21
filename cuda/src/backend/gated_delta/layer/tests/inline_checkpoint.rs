use super::{
    super::super::*,
    allocate, copy,
    packing::{dense_weights, pattern},
    read,
};
use crate::CudaConfig;

const TOKENS: usize = 40;
const BEFORE: usize = 24;

/// A checkpoint staged inside one pass must equal the state of a pass that
/// ended at the checkpoint, and must leave the outputs of the pass unchanged.
#[test]
fn staged_checkpoint_matches_a_pass_that_ends_there() -> Result<()> {
    for key_dim in [32, 128] {
        let backend = CudaBackend::new(CudaConfig::default())?;
        let (config, weights) = dense_weights(&backend, key_dim)?;
        let layer = CudaAffineGatedDeltaLayer::new(&backend, config, weights)?;
        let hidden = config.hidden_size;
        let values = pattern(TOKENS * hidden, 3);
        let probe = copy(&backend, &pattern(hidden, 5))?;

        let mut split_state = layer.prepare_state()?;
        let mut expected = Vec::new();
        for range in [0..BEFORE, BEFORE..TOKENS] {
            let input = copy(&backend, &values[range.start * hidden..range.end * hidden])?;
            let mut output = allocate(&backend, range.len() * hidden)?;
            layer.prepare(range.len())?.execute(&input, &mut split_state, &mut output)?;
            expected.extend(read(&backend, &output)?);
            if range.end == BEFORE {
                let mut restored = layer.prepare_state()?;
                restored.restore(&split_state.checkpoint()?)?;
                expected.extend(decode(&backend, &layer, &mut restored, &probe)?);
            }
        }

        let mut state = layer.prepare_state()?;
        state.arm_checkpoint(BEFORE);
        let mut output = allocate(&backend, TOKENS * hidden)?;
        layer
            .prepare(TOKENS)?
            .execute(&copy(&backend, &values)?, &mut state, &mut output)?;
        assert_eq!(state.offset(), TOKENS);
        let staged = state
            .take_staged_checkpoint()
            .ok_or(crate::Error::InvalidExecutionPlan("the pass staged no checkpoint"))?;
        let mut restored = layer.prepare_state()?;
        restored.restore(&staged)?;
        assert_eq!(restored.offset(), BEFORE);
        let output = read(&backend, &output)?;
        let mut actual = output[..BEFORE * hidden].to_vec();
        actual.extend(decode(&backend, &layer, &mut restored, &probe)?);
        actual.extend_from_slice(&output[BEFORE * hidden..]);

        assert!(expected.iter().any(|value| value.to_f32().abs() > 0.0));
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (actual.to_f32() - expected.to_f32()).abs() <= 0.001,
                "key_dim={key_dim} element {index}: {actual:?} != {expected:?}"
            );
        }
        assert_eq!(actual.len(), expected.len());
    }
    Ok(())
}

#[test]
fn ragged_rows_stage_only_the_armed_row() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let (config, weights) = dense_weights(&backend, 128)?;
    let layer = CudaAffineGatedDeltaLayer::new(&backend, config, weights)?;
    let hidden = config.hidden_size;
    let counts = [1, TOKENS, 7];
    let total = counts.iter().sum::<usize>();
    let input = copy(&backend, &pattern(total * hidden, 7))?;
    let run = |armed: bool| -> Result<(Vec<bf16>, bool)> {
        let mut states =
            (0..counts.len()).map(|_| layer.prepare_state()).collect::<Result<Vec<_>>>()?;
        if armed {
            states[1].arm_checkpoint(BEFORE);
        }
        let mut output = allocate(&backend, total * hidden)?;
        let mut rows = states.iter_mut().collect::<Vec<_>>();
        layer.prepare(total)?.execute_ragged(&input, &mut rows, &counts, &mut output)?;
        let staged = states[1].take_staged_checkpoint().is_some();
        assert!(states[0].take_staged_checkpoint().is_none());
        assert!(states[2].take_staged_checkpoint().is_none());
        Ok((read(&backend, &output)?, staged))
    };
    let (expected, unarmed) = run(false)?;
    let (actual, armed) = run(true)?;
    assert!(armed && !unarmed);
    for (actual, expected) in actual.iter().zip(&expected) {
        assert!((actual.to_f32() - expected.to_f32()).abs() <= 0.001);
    }
    Ok(())
}

fn decode(
    backend: &CudaBackend,
    layer: &CudaAffineGatedDeltaLayer,
    state: &mut CudaGatedDeltaState,
    probe: &DeviceBuffer<bf16>,
) -> Result<Vec<bf16>> {
    let mut output = allocate(backend, probe.len())?;
    layer.prepare(1)?.execute(probe, state, &mut output)?;
    read(backend, &output)
}
