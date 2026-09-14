use super::{Array, KvCache, PagedContextMode, Result, Stream, fixture};

#[test]
fn packed_native_attention_matches_rows_at_equal_and_unequal_positions() -> Result<()> {
    let mut stream = Stream::new_gpu()?;
    let attention = fixture(&stream)?;
    for (plan, history) in [
        (crate::config::RopeBatching::Rows, crate::config::HistoryBatching::Joined),
        (crate::config::RopeBatching::Offsets, crate::config::HistoryBatching::Joined),
        (crate::config::RopeBatching::Rows, crate::config::HistoryBatching::Rows),
        (crate::config::RopeBatching::Rows, crate::config::HistoryBatching::Gathered),
        (crate::config::RopeBatching::Rows, crate::config::HistoryBatching::Persistent),
    ] {
        stream.set_history_batching(history);
        stream.set_rope_batching(plan);
        for mode in [PagedContextMode::Native, PagedContextMode::View] {
            for lengths in [[128, 128], [128, 131]] {
                let mut caches = Vec::new();
                for length in lengths {
                    let mut cache = KvCache::new_paged(16, 16)?;
                    let input = Array::from_f32(
                        &vec![0.125; usize::try_from(length * 64)?],
                        &[1, length, 64],
                    )?;
                    let output = attention.forward(&input, &mut cache, 0, 0, true, &stream)?;
                    stream.eval_many_with_paged_arenas(&[&output])?;
                    caches.push(cache);
                }
                let mut packed = caches
                    .iter()
                    .map(|cache| cache.snapshot_at(cache.offset()?))
                    .collect::<Result<Vec<_>>>()?;
                let input = Array::from_f32(&vec![0.25; 128], &[2, 1, 64])?;
                let output = attention.forward_packed_decode_with_mode(
                    &input,
                    &mut packed.iter_mut().collect::<Vec<_>>(),
                    &lengths,
                    mode,
                    &stream,
                )?;
                let actual = output.to_vec_f32(&stream)?;
                for (row, cache) in caches.iter_mut().enumerate() {
                    let input = input.slice(&[row, 0, 0], &[row + 1, 1, 64], &stream)?;
                    let expected = attention
                        .forward_with_mode(&input, cache, 0, lengths[row], false, mode, &stream)?
                        .to_vec_f32(&stream)?;
                    let error = expected
                        .iter()
                        .zip(&actual[row * 64..(row + 1) * 64])
                        .map(|(left, right)| (left - right).abs())
                        .fold(0.0_f32, f32::max);
                    assert!(error < 1.0e-5, "lengths={lengths:?}, row={row}, error={error}");
                    assert_eq!(packed[row].offset()?, usize::try_from(lengths[row])? + 1);
                }
            }
        }
    }
    Ok(())
}
