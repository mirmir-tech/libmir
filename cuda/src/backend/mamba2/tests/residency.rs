//! Sequences moving between packed decode rows and solo execution keep
//! exactly the state they would have had running alone.

use mircuda::bf16;

use super::{Device, SMALL, copy, parameters, pattern, read, run, width};
use crate::{CudaConfig, Result, backend::CudaBackend};

#[test]
fn rows_survive_permutation_omission_and_solo_steps() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let device = Device::upload(&backend, &parameters(SMALL))?;
    let width = width(SMALL);
    let inner = SMALL.inner()?;
    let mut packed = Vec::new();
    let mut alone = Vec::new();
    for (index, tokens) in [3_usize, 5, 9].into_iter().enumerate() {
        let prompt = pattern(tokens * width, 20 + u64::try_from(index)?, 1.0);
        for states in [&mut packed, &mut alone] {
            let mut state = backend.prepare_mamba2_state(SMALL)?;
            run(&backend, &mut state, &prompt, &device)?;
            states.push(state);
        }
    }
    let mut batch = backend.prepare_mamba2_batch_state(SMALL, 3)?;
    // Steps: all three rows twice, then a permutation without sequence 1,
    // then sequence 1 alone, then all three again.
    let schedule: [&[usize]; 5] = [&[0, 1, 2], &[0, 1, 2], &[2, 0], &[1], &[0, 1, 2]];
    for (step, order) in schedule.into_iter().enumerate() {
        let projection = pattern(order.len() * width, 200 + u64::try_from(step)?, 1.0);
        let mut expected = Vec::new();
        for (row, &sequence) in order.iter().enumerate() {
            let rows = &projection[row * width..(row + 1) * width];
            expected.extend(run(&backend, &mut alone[sequence], rows, &device)?);
        }
        let actual = if let [sequence] = order {
            run(&backend, &mut packed[*sequence], &projection, &device)?
        } else {
            let mut selected = packed
                .iter_mut()
                .enumerate()
                .filter(|(index, _)| order.contains(index))
                .collect::<Vec<_>>();
            selected.sort_by_key(|(index, _)| order.iter().position(|value| value == index));
            let mut states = selected.into_iter().map(|(_, state)| state).collect::<Vec<_>>();
            batch.pack(&states)?;
            let input = copy(&backend, &projection)?;
            let mut output = backend
                .inner
                .pool
                .allocate::<bf16>(&backend.inner.stream, order.len() * inner)?;
            batch.decode(&input, device.view(), &mut output)?;
            batch.commit(&mut states)?;
            read(&backend, &output)?
        };
        assert_eq!(actual, expected, "step {step}");
    }
    for (sequence, (packed, alone)) in packed.iter().zip(&alone).enumerate() {
        let checkpoint = packed.checkpoint()?;
        let mut restored = backend.prepare_mamba2_state(SMALL)?;
        restored.restore(&checkpoint)?;
        assert_eq!(
            read(&backend, &restored.history)?,
            read(&backend, &alone.history)?,
            "history of sequence {sequence}"
        );
        assert_eq!(packed.offset(), alone.offset(), "offset of sequence {sequence}");
    }
    Ok(())
}
