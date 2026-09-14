use super::*;
use crate::engine::{Stream, lowering::MixerLowering};

fn linear_state() -> Result<SessionState> {
    Ok(SessionState::new(DecoderCache::new_hybrid_linear(&[MixerLowering::Linear], 1)?))
}

fn advance(state: &mut SessionState, stream: &Stream) -> Result<usize> {
    let value = Array::from_f32(&[1.0; 64], &[1, 1, 8, 8])?;
    let convolution = Array::from_f32(&[2.0; 12], &[1, 3, 4])?;
    state.cache.gated_delta_state(0)?.commit_compiled_decode(value, convolution);
    state.position += 1;
    let mut roots = Vec::new();
    state.cache.extend_graph_roots(&mut roots);
    stream.eval_many(&roots)?;
    stream.synchronize()?;
    Ok(state
        .cache
        .recurrent_allocations()?
        .iter()
        .map(mirtal::memory::Allocation::bytes)
        .sum())
}

#[test]
fn checkpoints_count_distinct_recurrent_generations_and_replacement() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let mut cache = PrefixCache::new(4, usize::MAX);
    let mut state = linear_state()?;
    let first = advance(&mut state, &stream)?;
    cache.insert_checkpoint("model", &[1], &state, 1, 100)?;
    assert_eq!(cache.resident_bytes(), 100 + first);
    let second = advance(&mut state, &stream)?;
    cache.insert_checkpoint("model", &[1, 2], &state, 1, 200)?;
    assert_eq!(cache.resident_bytes(), 200 + first + second);
    // Replacing a checkpoint and sharing it with a terminal does not add storage.
    cache.insert_checkpoint("model", &[1, 2], &state, 1, 200)?;
    let logits = Array::from_u32(&[1], &[1])?;
    cache.insert("model", &[1, 2], &state, &logits, None, 204)?;
    assert_eq!(cache.resident_bytes(), 204 + first + second);
    cache.clear();
    assert_eq!(cache.resident_bytes(), 0);
    Ok(())
}

#[test]
fn byte_limit_evicts_recurrent_state_even_when_kv_estimate_fits() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let mut state = linear_state()?;
    let bytes = advance(&mut state, &stream)?;
    let mut next = linear_state()?;
    let next_bytes = advance(&mut next, &stream)?;
    // Allocator reuse can give equal-shaped arrays different backing capacities.
    // Either entry must fit independently, while both together exceed the limit.
    let mut cache = PrefixCache::new(4, 100 + bytes.max(next_bytes));
    cache.insert_checkpoint("first", &[1], &state, 1, 100)?;
    assert_eq!(cache.resident_bytes(), 100 + bytes);
    assert!(cache.lease_longest("first", &[1, 3])?.is_some());
    cache.insert_checkpoint("second", &[2], &next, 1, 100)?;
    assert_eq!(cache.group_count(), 1);
    assert_eq!(cache.resident_bytes(), 100 + next_bytes);
    assert!(cache.lease_longest("first", &[1, 3])?.is_none());
    assert!(cache.lease_longest("second", &[2, 3])?.is_some());
    Ok(())
}

#[test]
fn counts_shared_parent_across_groups_until_last_row_is_evicted() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let parent = Array::from_f32(&[1.0; 128], &[2, 1, 8, 8])?;
    let history = Array::from_f32(&[2.0; 24], &[2, 3, 4])?;
    let mut bytes = 0;
    let mut cache = PrefixCache::new(4, usize::MAX);
    for row in 0..2 {
        let mut state = linear_state()?;
        state.cache.gated_delta_state(0)?.commit_compiled_decode(
            parent.slice(&[row, 0, 0, 0], &[row + 1, 1, 8, 8], &stream)?,
            history.slice(&[row, 0, 0], &[row + 1, 3, 4], &stream)?,
        );
        state.position = 1;
        let mut roots = Vec::new();
        state.cache.extend_graph_roots(&mut roots);
        stream.eval_many(&roots)?;
        stream.synchronize()?;
        if row == 0 {
            bytes = state
                .cache
                .recurrent_allocations()?
                .iter()
                .map(mirtal::memory::Allocation::bytes)
                .sum();
            assert!(bytes >= (128 + 24) * size_of::<f32>());
        }
        cache.insert_checkpoint(
            if row == 0 {
                "first"
            } else {
                "second"
            },
            &[1],
            &state,
            1,
            100,
        )?;
    }
    assert_eq!(cache.resident_bytes(), 200 + bytes);
    assert!(cache.evict_oldest());
    assert_eq!(cache.resident_bytes(), 100 + bytes, "a single row retains the whole parent");
    assert!(cache.evict_oldest());
    assert_eq!(cache.resident_bytes(), 0);
    Ok(())
}

#[test]
fn exact_recurrent_checkpoint_replays_from_an_earlier_usable_state() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let mut state = linear_state()?;
    let mut cache = PrefixCache::new(4, usize::MAX);
    advance(&mut state, &stream)?;
    cache.insert_checkpoint("model", &[1], &state, 1, 100)?;
    assert!(cache.lease_longest("model", &[1])?.is_none());
    advance(&mut state, &stream)?;
    cache.insert_checkpoint("model", &[1, 2], &state, 1, 200)?;
    let restored = cache
        .lease_longest("model", &[1, 2])?
        .ok_or(crate::native::error::Error::NoPrefixLogits)?;
    assert_eq!(restored.restored.0.position, 1);
    assert!(restored.restored.1.is_none());
    Ok(())
}
