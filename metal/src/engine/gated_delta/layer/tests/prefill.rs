use super::*;

#[test]
fn packed_prefill_preserves_rows_snapshots_and_decode_continuation() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let mut layer = layer(&stream)?;
    layer.in_proj_qkv = varying_linear(192, &stream)?;
    layer.in_proj_z = varying_linear(64, &stream)?;
    layer.out_proj = varying_linear(64, &stream)?;
    layer.conv_weight = Array::from_f32(&vec![0.25; 384], &[192, 2, 1])?;
    layer.compiled_decode = CompiledDecode::new(&layer, &stream)?;
    let mut packed = (0..3).map(|_| GatedDeltaState::new()).collect::<Result<Vec<_>>>()?;
    let mut rows = (0..3).map(|_| GatedDeltaState::new()).collect::<Result<Vec<_>>>()?;
    let mut saved = Vec::new();
    for sequence in [3, 7, 2, 9] {
        // Reorder an existing cohort, then shrink it: both must join actual views.
        if sequence == 7 {
            packed.rotate_left(1);
            rows.rotate_left(1);
        }
        if sequence == 2 {
            packed.pop();
            rows.pop();
        }
        let input = Array::from_f32(
            &(0..packed.len() * sequence * 64)
                .map(|i| Ok(f32::from(i16::try_from(i % 31)?) * 0.01 - 0.15))
                .collect::<Result<Vec<_>>>()?,
            &[i32::try_from(packed.len())?, i32::try_from(sequence)?, 64],
        )?;
        let actual = required(
            layer.forward_packed_prefill(
                &input,
                &mut packed.iter_mut().collect::<Vec<_>>(),
                &stream,
            )?,
            "packed prefill",
        )?;
        let mut expected = Vec::new();
        for (row, state) in rows.iter_mut().enumerate() {
            let input = input.slice(&[row, 0, 0], &[row + 1, sequence, 64], &stream)?;
            expected.push(layer.forward(&input, state, &stream)?);
        }
        let expected = Array::concatenate(&expected.iter().collect::<Vec<_>>(), 0, &stream)?;
        compare(&actual, &expected, &stream)?;
        for (a, b) in packed.iter().zip(&rows) {
            compare(&a.values()?, &b.values()?, &stream)?;
            compare(
                required(a.convolution.as_deref(), "history")?,
                required(b.convolution.as_deref(), "reference history")?,
                &stream,
            )?;
            assert_eq!(a.offset()?, b.offset()?);
            stream.eval_many(&a.graph_roots().collect::<Vec<_>>())?;
            stream.synchronize()?;
            a.detach_evaluated_graphs(&stream)?;
        }
        if sequence == 3 {
            for state in &packed {
                saved.push((state.snapshot()?, state.values()?.to_vec_f32(&stream)?));
            }
        }
    }
    let input =
        Array::from_f32(&vec![0.02; packed.len() * 64], &[i32::try_from(packed.len())?, 1, 64])?;
    let actual = required(
        layer.forward_packed(&input, &mut packed.iter_mut().collect::<Vec<_>>(), &stream)?,
        "packed decode",
    )?;
    let expected = required(
        layer.forward_packed(&input, &mut rows.iter_mut().collect::<Vec<_>>(), &stream)?,
        "reference decode",
    )?;
    compare(&actual, &expected, &stream)?;
    for (state, values) in saved {
        assert_eq!(state.values()?.to_vec_f32(&stream)?, values);
    }
    // An incomplete generation is rejected before mutating any row.
    packed[1].reset()?;
    let saved_offset = packed[0].offset()?;
    let input = Array::from_f32(&vec![0.02; 2 * 2 * 64], &[2, 2, 64])?;
    assert!(
        layer
            .forward_packed_prefill(&input, &mut packed.iter_mut().collect::<Vec<_>>(), &stream)?
            .is_none()
    );
    assert_eq!(packed[0].offset()?, saved_offset);
    assert_eq!(packed[1].offset()?, 0);
    Ok(())
}

fn compare(actual: &Array, expected: &Array, stream: &Stream) -> Result<()> {
    let a = actual.to_vec_f32(stream)?;
    let b = expected.to_vec_f32(stream)?;
    assert_eq!(actual.shape()?, expected.shape()?);
    assert!(a.iter().zip(&b).all(|(a, b)| (a - b).abs() <= 1e-5));
    assert!(a.iter().any(|value| *value != 0.0));
    Ok(())
}
