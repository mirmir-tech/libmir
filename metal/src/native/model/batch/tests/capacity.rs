use std::sync::Arc;

use super::*;
use crate::engine::{Array, DecoderCache, KvCache, KvPageFormat};

#[test]
fn rejects_aggregate_decode_demand_before_consuming_any_row() -> Result<()> {
    exercise(Case::Packed)
}

#[test]
fn rejects_mixed_group_before_consuming_the_zero_forward_row() -> Result<()> {
    exercise(Case::Mixed)
}

#[test]
fn scalar_rejection_preserves_pending_token_for_retry() -> Result<()> {
    exercise(Case::Scalar)
}

#[derive(Clone, Copy)]
enum Case {
    Packed,
    Mixed,
    Scalar,
}

fn exercise(case: Case) -> Result<()> {
    let mut config = crate::MetalConfig::default();
    config.cache.prefix_cache_entries = 0;
    config.cache.paged_attention_min_context = 1;
    config.kv_cache.block_count = 1;
    config.set_max_batch_requests(1);
    config.tuning.mode = runtime::tuning::TuningMode::Disabled;
    let (mut model, directory) = fixture::load_config(config)?;
    let mut inputs = Vec::new();
    for seed in [0, 1] {
        let session = Uuid::new_v4();
        let prompt = (0..31).map(|index| (index + seed) % 63 + 1).collect::<Vec<_>>();
        let output =
            model.prefill(session, &prompt, &[], SamplingLogits::None, None, &mut |_| {})?;
        let mut token = choose(&model, output.output, SamplingLogits::None)?;
        for _ in 0..16 {
            let output = model.decode(session, token, SamplingLogits::None)?;
            token = choose(&model, output, SamplingLogits::None)?;
        }
        assert_eq!(model.sessions[&session].cache.cached_tokens()?, 48);
        inputs.push(DecodeInput {
            session,
            token,
            sampling: SamplingLogits::None,
        });
    }
    model.flush_decode_graphs()?;
    let maximum = DecoderCache::physical_page_capacity(model.stream(), model.info.cache_step);
    let pool = Arc::clone(model.stream().paged_arenas());
    let available = pool.available_pages(maximum, 0, 2, 8, KvPageFormat::Native)?;
    let remaining = usize::from(matches!(case, Case::Packed));
    let count = available - remaining;
    if matches!(case, Case::Mixed) {
        inputs[0].sampling = SamplingLogits::Full;
    }
    let mut blocker = KvCache::new_paged_with_pool_capacity(
        16,
        16,
        KvPageFormat::Native,
        maximum,
        Arc::clone(&pool),
        0,
    )?;
    let keys =
        Array::from_f32(&vec![0.0; 2 * count * 16 * 8], &[1, 2, i32::try_from(count * 16)?, 8])?;
    drop(blocker.update(&keys, &keys, model.stream())?);
    pool.eval_with_graph_roots(&[], model.stream())?;
    assert_eq!(pool.available_pages(maximum, 0, 2, 8, KvPageFormat::Native)?, remaining);
    let rejected = if matches!(case, Case::Scalar) {
        model.decode(inputs[0].session, inputs[0].token, inputs[0].sampling).map(|_| ())
    } else {
        model.decode_grouped(&inputs).map(|_| ())
    };
    assert!(matches!(rejected, Err(crate::native::error::Error::Engine(
        crate::engine::Error::KvPageCapacity { layer: 0, required, maximum: limit }
    )) if required == maximum + 1 && limit == maximum));
    for input in &inputs {
        let state = &model.sessions[&input.session];
        assert_eq!(state.position, 47, "rejected group consumed pending state");
        assert_eq!(state.cache.cached_tokens()?, 48);
        assert_eq!(state.pending.as_ref().map(|pending| pending.token_id), Some(input.token));
    }
    assert_eq!(pool.available_pages(maximum, 0, 2, 8, KvPageFormat::Native)?, remaining);
    blocker.reset()?;
    let references = inputs
        .iter()
        .map(|input| {
            let reference = Uuid::new_v4();
            let state = model.sessions[&input.session].snapshot()?;
            model.sessions.insert(reference, state);
            Ok(reference)
        })
        .collect::<Result<Vec<_>>>()?;
    for _ in 0..3 {
        let outputs = model.decode_grouped(&inputs)?;
        assert_eq!(outputs.len(), 2);
        for ((input, reference), (output, _)) in inputs.iter_mut().zip(&references).zip(outputs) {
            let expected = model.decode(*reference, input.token, SamplingLogits::Full)?;
            let expected = choose(&model, expected, SamplingLogits::Full)?;
            input.token = choose(&model, output, input.sampling)?;
            assert_eq!(input.token, expected, "retry changed continuation");
            input.sampling = SamplingLogits::None;
        }
    }
    for input in inputs {
        assert_eq!(model.sessions[&input.session].cache.cached_tokens()?, 51);
        model.release_session(input.session)?;
    }
    for reference in references {
        model.release_session(reference)?;
    }
    assert_eq!(pool.resident_arenas()?, 0);
    drop(model);
    std::fs::remove_dir_all(directory)?;
    Ok(())
}
