use runtime::kv::{CacheConfig, KvCache, KvCacheDType};

use super::*;

#[test]
fn cache_pressure_retry_appends_the_decode_token_once() -> Result<()> {
    let mut cache = KvCache::with_config(CacheConfig {
        block_size: 2,
        block_count: 3,
        dtype: KvCacheDType::Auto,
    });
    let mut state = KvSessionState::new(Uuid::new_v4(), "model", 2);
    let mut busy = KvSessionState::new(Uuid::new_v4(), "model", 2);
    state.append_decode_in_place(&mut cache, 4)?;
    busy.reserve_capacity_in_place(&mut cache, 4)?;
    for _ in 0..2 {
        assert!(matches!(
            reserve_decode(&mut state, &mut cache, 5, Some(1)),
            Err(crate::Error::Runtime(runtime::RuntimeError::KvCachePressure))
        ));
        assert_eq!(state.token_len(), 1);
        assert_eq!(state.table().token_len(), 1);
    }
    busy.release(&mut cache)?;
    let step = reserve_decode(&mut state, &mut cache, 5, Some(1))?;
    assert_eq!(step, 1);
    assert_eq!(state.token_len(), 2);
    assert_eq!(state.table().token_len(), 2);
    assert_eq!(state.commit_ready_prefix_blocks(&mut cache)?, 1);
    state.release(&mut cache)?;
    Ok(())
}
