use super::*;
mod budget;
mod reclamation;

fn data(
    batch: usize,
    heads: usize,
    tokens: usize,
    dim: usize,
    stream: &Stream,
    dtype: mirtal::DType,
) -> Result<Array> {
    let values = (0..batch * heads * tokens * dim)
        .map(|i| Ok((f32::from(u16::try_from(i % 29)?) - 14.0) / 32.0))
        .collect::<Result<Vec<_>>>()?;
    let array = Array::from_f32(
        &values,
        &[
            i32::try_from(batch)?,
            i32::try_from(heads)?,
            i32::try_from(tokens)?,
            i32::try_from(dim)?,
        ],
    )?;
    Array::from_native(stream.native().graph().astype(array.native(), dtype)?)
}

#[test]
fn append_preserves_lazy_readers_strided_inputs_and_allocation_until_growth() -> Result<()> {
    let stream = Stream::new_gpu()?;
    for dtype in [mirtal::DType::Float32, mirtal::DType::Float16, mirtal::DType::Bfloat16] {
        for (heads, dim) in [(2, 256), (8, 64)] {
            let full = data(3, heads, 258, dim, &stream, dtype)?;
            let query = data(3, heads * 2, 1, dim, &stream, dtype)?;
            let mut previous = None;
            let mut retained = Vec::new();
            let mut outputs = Vec::new();
            for tokens in 254..=258 {
                let contexts = (0..3)
                    .map(|row| {
                        Ok(KvContext {
                            keys: full.slice(
                                &[row, 0, 0, 0],
                                &[row + 1, heads, tokens, dim],
                                &stream,
                            )?,
                            values: full.slice(
                                &[row, 0, 0, 0],
                                &[row + 1, heads, tokens, dim],
                                &stream,
                            )?,
                            paged: None,
                            mask: None,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                let update =
                    full.slice(&[0, 0, tokens - 1, 0], &[3, heads, tokens, dim], &stream)?;
                let next =
                    History::next(previous.as_ref(), &contexts, [&update, &update], &stream)?;
                let [keys, values] = next.views(&stream)?;
                let actual =
                    query.scaled_dot_product_attention(&keys, &values, 0.125, false, &stream)?;
                let joined = Array::concatenate(
                    &contexts.iter().map(|c| &c.keys).collect::<Vec<_>>(),
                    0,
                    &stream,
                )?;
                let expected =
                    query.scaled_dot_product_attention(&joined, &joined, 0.125, false, &stream)?;
                outputs.push((actual, expected));
                retained.push(next.buffers[0].snapshot()?);
                previous = Some(next);
            }
            // Evaluate newest first: an older view must survive all later writes
            // even when its SDPA has not yet been submitted.
            for (actual, expected) in outputs.iter().rev() {
                assert_eq!(actual.to_vec_f32(&stream)?, expected.to_vec_f32(&stream)?);
            }
            let allocations = retained
                .iter()
                .map(|a| a.native().allocation())
                .collect::<mirtal::Result<Vec<_>>>()?;
            assert!(allocations.iter().all(Option::is_some));
            assert_eq!(allocations[0], allocations[1]);
            assert_eq!(allocations[1], allocations[2]);
            assert_ne!(allocations[2], allocations[3]);
            assert_eq!(allocations[3], allocations[4]);
        }
    }
    Ok(())
}

fn attach(caches: &mut [KvCache], stream: &Stream) -> Result<()> {
    let full = data(2, 2, 1, 64, stream, mirtal::DType::Float32)?;
    let contexts = (0..2)
        .map(|row| {
            Ok(KvContext {
                keys: full.slice(&[row, 0, 0, 0], &[row + 1, 2, 1, 64], stream)?,
                values: full.slice(&[row, 0, 0, 0], &[row + 1, 2, 1, 64], stream)?,
                paged: None,
                mask: None,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let history = Arc::new(History::next(None, &contexts, [&full, &full], stream)?);
    for (index, cache) in caches.iter_mut().enumerate() {
        cache.history = Some(Row { index, history: Arc::clone(&history) });
    }
    Ok(())
}

#[test]
fn cohort_ownership_rejects_reorder_removal_replacement_snapshot_and_other_updates() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let mut caches = [KvCache::new(16)?, KvCache::new(16)?];
    attach(&mut caches, &stream)?;
    assert!(take(&mut caches.iter_mut().collect::<Vec<_>>()).is_some());
    assert!(caches.iter().all(|c| c.history.is_none()));
    attach(&mut caches, &stream)?;
    caches.swap(0, 1);
    assert!(take(&mut caches.iter_mut().collect::<Vec<_>>()).is_none());
    attach(&mut caches, &stream)?;
    assert!(take(&mut [&mut caches[0]]).is_none());
    attach(&mut caches, &stream)?;
    caches[1] = KvCache::new(16)?;
    assert!(take(&mut caches.iter_mut().collect::<Vec<_>>()).is_none());
    attach(&mut caches, &stream)?;
    assert!(caches[0].snapshot_at(0)?.history.is_none());
    caches[0].reset()?;
    assert!(take(&mut caches.iter_mut().collect::<Vec<_>>()).is_none());
    attach(&mut caches, &stream)?;
    let update = data(1, 2, 1, 64, &stream, mirtal::DType::Float32)?;
    caches[1].update(&update, &update, &stream)?;
    assert!(take(&mut caches.iter_mut().collect::<Vec<_>>()).is_none());
    Ok(())
}

#[test]
fn rejects_mismatched_updates_before_aliasing() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let full = data(1, 2, 2, 64, &stream, mirtal::DType::Float32)?;
    let contexts = [KvContext {
        keys: full.snapshot()?,
        values: full.snapshot()?,
        paged: None,
        mask: None,
    }];
    assert!(History::next(None, &contexts, [&full, &full], &stream).is_err());
    Ok(())
}
