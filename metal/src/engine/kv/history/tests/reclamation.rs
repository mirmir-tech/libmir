use super::*;
use crate::engine::Error;

#[test]
fn reservation_reclamation_releases_all_cohort_owners_but_preserves_pending_readers() -> Result<()>
{
    let stream = Stream::new_gpu()?;
    let mut caches = [KvCache::new(16)?, KvCache::new(16)?];
    attach(&mut caches, &stream)?;
    let weak = Arc::downgrade(
        &caches[0]
            .history
            .as_ref()
            .ok_or(Error::NullHandle("test cohort history"))?
            .history,
    );
    let [keys, values] = caches[0]
        .history
        .as_ref()
        .ok_or(Error::NullHandle("test cohort history"))?
        .history
        .views(&stream)?;
    let query = data(2, 2, 1, 64, &stream, mirtal::DType::Float32)?;
    let output = query.scaled_dot_product_attention(&keys, &values, 0.125, false, &stream)?;
    let expected = values.to_vec_f32(&stream)?;
    caches[0].release_reservation_after(0)?;
    assert!(weak.upgrade().is_some());
    caches[1].release_reservation_after(0)?;
    assert!(weak.upgrade().is_none());
    assert!(take(&mut caches.iter_mut().collect::<Vec<_>>()).is_none());
    assert_eq!(output.to_vec_f32(&stream)?, expected);
    Ok(())
}

#[test]
#[ignore = "isolated allocator measurement; run without other tests"]
fn prepared_plan_does_not_retain_reclaimed_buffers() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let mut caches = [KvCache::new(16)?, KvCache::new(16)?];
    attach(&mut caches, &stream)?;
    let previous = take(&mut caches.iter_mut().collect::<Vec<_>>())
        .ok_or(Error::NullHandle("test cohort history"))?;
    let update = data(2, 2, 1, 64, &stream, mirtal::DType::Float32)?;
    let buffers =
        stream
            .kernels()
            .history_append
            .execute(&previous, [&update, &update], &stream)?;
    let next = Arc::new(History {
        buffers,
        shape: previous.shape,
        tokens: 2,
        reservation: Arc::clone(&previous.reservation),
    });
    stream.eval_many(&next.buffers.iter().collect::<Vec<_>>())?;
    stream.synchronize()?;
    // Allocation inventory retains storage, so discard it before reclamation.
    let retained_bytes = next
        .buffers
        .iter()
        .map(|b| b.native().allocation()?.ok_or(Error::NullHandle("test allocation")))
        .collect::<Result<std::collections::HashSet<_>>>()?
        .iter()
        .map(mirtal::memory::Allocation::bytes)
        .sum::<usize>();
    for (index, cache) in caches.iter_mut().enumerate() {
        cache.history = Some(Row { index, history: Arc::clone(&next) });
    }
    drop(next);
    drop(previous);
    let before = crate::engine::memory_stats()?.active;
    for cache in &mut caches {
        cache.release_reservation_after(0)?;
    }
    let after = crate::engine::memory_stats()?.active;
    assert!(
        before.saturating_sub(after) >= retained_bytes,
        "before={before}, after={after}, buffers={retained_bytes}"
    );
    Ok(())
}
