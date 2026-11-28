use runtime::{backend::SamplingLogits, kv::BlockId};
use uuid::Uuid;

use super::{DecodeInput, fixture};
use crate::native::error::{Error, Result};

#[test]
fn rejects_out_of_range_block_id_without_advancing_any_session() -> Result<()> {
    let (mut model, directory) = fixture::load()?;
    let capacity = model.stream().config().kv_cache.block_count;
    let first = Uuid::new_v4();
    let second = Uuid::new_v4();
    let prompt = [1_u32, 2, 3, 4];
    let first_out =
        model.prefill(first, &prompt, &[], &SamplingLogits::None, Some(16), &mut |_| {})?;
    let second_out =
        model.prefill(second, &prompt, &[], &SamplingLogits::None, Some(16), &mut |_| {})?;
    let first_token = super::choose(&model, first_out.output, &SamplingLogits::None)?;
    let second_token = super::choose(&model, second_out.output, &SamplingLogits::None)?;
    let before = model.sessions[&first].position;

    let mut valid = DecodeInput::single(first, first_token, SamplingLogits::None);
    valid.blocks = vec![BlockId(0)];
    let mut invalid = DecodeInput::single(second, second_token, SamplingLogits::None);
    invalid.blocks = vec![BlockId(capacity)];

    let error = model.decode_grouped(&[valid, invalid]);
    assert!(matches!(
        error,
        Err(Error::Engine(crate::engine::Error::KvBlockOutOfRange { block, .. }))
            if block == capacity
    ));
    let runtime_error: runtime::RuntimeError = error
        .err()
        .ok_or_else(|| Error::InvalidDecodeBatch("expected out-of-range block error".into()))?
        .into();
    assert!(matches!(runtime_error, runtime::RuntimeError::KvCapacity { .. }));
    assert_eq!(model.sessions[&first].position, before);
    assert_eq!(model.sessions[&second].position, before);

    model.release_session(first)?;
    model.release_session(second)?;
    drop(model);
    std::fs::remove_dir_all(directory)?;
    Ok(())
}
