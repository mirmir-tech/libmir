use super::*;
use crate::engine::DecoderCache;

#[test]
fn impossible_scalar_prefill_preserves_clamped_full_prefix_cache() -> Result<()> {
    exercise(Family::Clamped)
}

#[test]
fn impossible_scalar_prefill_preserves_hybrid_full_prefix_cache() -> Result<()> {
    exercise(Family::Hybrid)
}

fn exercise(family: Family) -> Result<()> {
    let (mut model, directory) = load_family(family, 1)?;
    let prompt = (0..32).collect::<Vec<u32>>();
    let session = Uuid::new_v4();
    let first = model.prefill(session, &prompt, &[], SamplingLogits::Full, None, &mut |_| {})?;
    let expected = match first.output {
        NativeOutput::Logits(logits) => logits.to_vec_f32(model.stream())?,
        NativeOutput::Greedy(_) => {
            return Err(Error::InvalidPrefillBatch("missing reference logits".into()));
        },
    };
    model.release_session(session)?;
    let maximum = DecoderCache::physical_page_capacity(model.stream(), model.info.cache_step);
    let impossible = vec![3; maximum * model.stream().config().kv_cache.block_size + 1];
    let rejected_session = Uuid::new_v4();
    let mut progressed = false;
    let rejected =
        model.prefill(rejected_session, &impossible, &[], SamplingLogits::Full, None, &mut |_| {
            progressed = true;
        });
    assert!(rejected.is_err());
    assert!(!progressed);
    assert!(model.sessions.is_empty());
    assert_eq!(
        model.prefixes.group_count(),
        1,
        "failed scalar prefill evicted a reusable prefix"
    );
    let session = Uuid::new_v4();
    let restored = model.prefill(session, &prompt, &[], SamplingLogits::Full, None, &mut |_| {})?;
    assert_eq!(restored.prefix_cache_tokens, prompt.len());
    let actual = match restored.output {
        NativeOutput::Logits(logits) => logits.to_vec_f32(model.stream())?,
        NativeOutput::Greedy(_) => {
            return Err(Error::InvalidPrefillBatch("missing restored logits".into()));
        },
    };
    assert_eq!(actual, expected);
    model.release_session(session)?;
    model.clear_prefix_cache();
    assert_eq!(model.stream().paged_arenas().resident_arenas()?, 0);
    drop(model);
    std::fs::remove_dir_all(directory)?;
    Ok(())
}
