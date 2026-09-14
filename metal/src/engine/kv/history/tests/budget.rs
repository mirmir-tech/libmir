use super::*;
use crate::engine::Error;

#[test]
fn append_shares_charge_and_failed_growth_does_not_allocate_or_change_history() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let full = data(2, 2, 257, 64, &stream, mirtal::DType::Float32)?;
    let make = |tokens| -> Result<(Vec<KvContext>, Array)> {
        let contexts = (0..2)
            .map(|row| {
                Ok(KvContext {
                    keys: full.slice(&[row, 0, 0, 0], &[row + 1, 2, tokens, 64], &stream)?,
                    values: full.slice(&[row, 0, 0, 0], &[row + 1, 2, tokens, 64], &stream)?,
                    mask: None,
                    paged: None,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok((contexts, full.slice(&[0, 0, tokens - 1, 0], &[2, 2, tokens, 64], &stream)?))
    };
    let bytes = super::super::budget::storage_bytes([2, 2, 256, 64], mirtal::DType::Float32)?;
    stream.history_budget().set_limit(bytes);
    let (contexts, update) = make(255)?;
    let first = History::next(None, &contexts, [&update, &update], &stream)?;
    let (contexts, update) = make(256)?;
    let second = History::next(Some(&first), &contexts, [&update, &update], &stream)?;
    assert_eq!(stream.history_budget().snapshot().retained_bytes, bytes);
    let (contexts, update) = make(257)?;
    assert!(matches!(
        History::next(Some(&second), &contexts, [&update, &update], &stream),
        Err(Error::HistoryBudgetUnavailable)
    ));
    assert_eq!(stream.history_budget().snapshot().retained_bytes, bytes);
    let prior = second.views(&stream)?;
    assert_eq!(prior[0].shape()?, [2, 2, 256, 64]);
    drop(first);
    assert_eq!(stream.history_budget().snapshot().retained_bytes, bytes);
    drop(second);
    assert_eq!(stream.history_budget().snapshot().retained_bytes, 0);
    // Readers can outlive cache ownership; the quota intentionally counts only
    // retained history objects, not every MLX graph or allocator allocation.
    assert_eq!(prior[0].to_vec_f32(&stream)?.len(), 2 * 2 * 256 * 64);
    stream.history_budget().set_limit(bytes * 2);
    let grown = History::next(None, &contexts, [&update, &update], &stream)?;
    assert_eq!(grown.shape, [2, 2, 512, 64]);
    assert_eq!(stream.history_budget().snapshot().retained_bytes, bytes * 2);
    Ok(())
}

#[test]
fn zero_budget_uses_joined_attention_without_retaining_history() -> Result<()> {
    let stream = Stream::new_gpu()?;
    stream.history_budget().set_limit(0);
    let mut caches = [KvCache::new(16)?, KvCache::new(16)?];
    let update = data(2, 2, 1, 64, &stream, mirtal::DType::Float32)?;
    let contexts = (0..2)
        .map(|row| {
            Ok(KvContext {
                keys: update.slice(&[row, 0, 0, 0], &[row + 1, 2, 1, 64], &stream)?,
                values: update.slice(&[row, 0, 0, 0], &[row + 1, 2, 1, 64], &stream)?,
                mask: None,
                paged: None,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let output = execute(
        None,
        &mut caches.iter_mut().collect::<Vec<_>>(),
        &contexts,
        [&update, &update],
        &update,
        0.125,
        &stream,
    )?;
    assert_eq!(output.to_vec_f32(&stream)?, update.to_vec_f32(&stream)?);
    assert!(caches.iter().all(|c| c.history.is_none()));
    assert_eq!(stream.history_budget().snapshot().retained_bytes, 0);
    Ok(())
}
