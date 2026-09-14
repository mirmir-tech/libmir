use super::*;

#[test]
fn compacted_history_preserves_chunked_convolution_and_snapshots() -> Result<()> {
    let stream = Stream::new_gpu()?;
    for rows in [1, 3] {
        let input = Array::from_f32(
            &(0..rows * 512 * 64)
                .map(|index| Ok(f32::from(u16::try_from(index % 127)?) / 127.0))
                .collect::<Result<Vec<_>>>()?,
            &[rows, 512, 64],
        )?;
        let weight = Array::from_f32(&[0.25; 64 * 4], &[64, 4, 1])?;
        let mut state = GatedDeltaState::new()?;
        let output = convolve(&mut state, &input, &weight, &stream)?;
        compact_history(&mut state, &stream)?;
        let history = state.convolution_root().ok_or_else(missing_history)?;
        stream.eval_many(&[&output, history])?;
        stream.synchronize()?;
        state.detach_evaluated_graphs(&stream)?;
        let allocation = history.native().allocation()?.ok_or_else(missing_history)?;
        let logical_bytes = usize::try_from(rows)? * 3 * 64 * size_of::<f32>();
        assert!(allocation.bytes() <= logical_bytes + 16384);
        let expected = input.slice(&[0, 509, 0], &[usize::try_from(rows)?, 512, 64], &stream)?;
        assert_eq!(history.to_vec_f32(&stream)?, expected.to_vec_f32(&stream)?);

        // A retained snapshot must survive advancement and reproduce the same
        // suffix as a single convolution over the entire sequence.
        let saved = state.snapshot()?;
        let next = Array::from_f32(&vec![0.5; usize::try_from(rows)? * 9 * 64], &[rows, 9, 64])?;
        let suffix = convolve(&mut state, &next, &weight, &stream)?;
        let joined = Array::concatenate(&[&input, &next], 1, &stream)?;
        let mut reference = GatedDeltaState::new()?;
        let whole = convolve(&mut reference, &joined, &weight, &stream)?;
        let suffix_reference =
            whole.slice(&[0, 512, 0], &[usize::try_from(rows)?, 521, 64], &stream)?;
        assert_eq!(suffix.to_vec_f32(&stream)?, suffix_reference.to_vec_f32(&stream)?);
        assert_eq!(
            saved.convolution_root().ok_or_else(missing_history)?.to_vec_f32(&stream)?,
            expected.to_vec_f32(&stream)?,
        );
    }
    Ok(())
}

// Prepare retention before the existing materialization, not after every chunk.
fn compact_history(state: &mut GatedDeltaState, stream: &Stream) -> Result<()> {
    state.prepare_prefix_retention(stream)
}

fn missing_history() -> Error {
    Error::InvalidModel("expected materialized convolution history".into())
}
