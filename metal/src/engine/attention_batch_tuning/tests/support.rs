use crate::engine::{Array, KvCache, KvContext, PagedContextMode, Result, Stream};

pub(super) fn paged_context(
    tokens: usize,
    head_dim: usize,
    seed: usize,
    stream: &Stream,
) -> Result<KvContext> {
    let values = (0..tokens * head_dim).map(|index| patterned(index, seed)).collect::<Vec<_>>();
    let shape = [1, 1, i32::try_from(tokens)?, i32::try_from(head_dim)?];
    let keys = Array::from_f32(&values, &shape)?;
    let values = Array::from_f32(&values.iter().rev().copied().collect::<Vec<_>>(), &shape)?;
    let mut cache = KvCache::new_paged(tokens, 16)?;
    cache.update_for_attention_mode(&keys, &values, stream, 0, PagedContextMode::Both)
}

pub(super) fn native_decode_context(
    tokens: usize,
    head_dim: usize,
    seed: usize,
    stream: &Stream,
) -> Result<KvContext> {
    let prefix = tokens - 1;
    let values = (0..prefix * head_dim).map(|index| patterned(index, seed)).collect::<Vec<_>>();
    let shape = [1, 1, i32::try_from(prefix)?, i32::try_from(head_dim)?];
    let keys = Array::from_f32(&values, &shape)?;
    let mut cache = KvCache::new_paged(tokens, 16)?;
    drop(cache.update_for_attention_mode(&keys, &keys, stream, 0, PagedContextMode::View)?);
    let token = Array::from_f32(&vec![0.5; head_dim], &[1, 1, 1, i32::try_from(head_dim)?])?;
    cache.update_for_attention_mode(&token, &token, stream, 0, PagedContextMode::Native)
}

pub(super) fn patterned(index: usize, seed: usize) -> f32 {
    u8::try_from((index * seed + 3) % 101).map_or(0.0, f32::from) / 50.0 - 1.0
}

pub(super) fn assert_outputs_close(
    expected: &[Array],
    actual: &[Array],
    stream: &Stream,
) -> Result<()> {
    let expected = Array::concatenate(&expected.iter().collect::<Vec<_>>(), 0, stream)?;
    let actual = Array::concatenate(&actual.iter().collect::<Vec<_>>(), 0, stream)?;
    actual.async_eval()?;
    stream.synchronize()?;
    let expected = expected.to_vec_f32()?;
    let actual = actual.to_vec_f32()?;
    assert_eq!(expected.len(), actual.len());
    assert!(expected.iter().zip(actual).all(|(left, right)| (left - right).abs() < 1.0e-4));
    Ok(())
}
