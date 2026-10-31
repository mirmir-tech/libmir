use super::{
    fixture::{Family, load_family},
    *,
};
use crate::config::PrefillMaterialization::{OutputAndState, StateOnly};

#[test]
fn state_only_prefill_preserves_batched_logits_continuation_and_prefixes() -> Result<()> {
    for (family, page_min_context) in
        [(Family::Hybrid, 1), (Family::Clamped, 1), (Family::Clamped, 128), (Family::Hybrid, 128)]
    {
        let (mut model, directory) =
            fixture::load_family_with_paging(family, 64, page_min_context)?;
        model.info.prefill_step = 16;
        for (width, budget) in [(1, 16), (3, 48), (5, 17)] {
            let mut reference = None;
            for mode in [OutputAndState, StateOnly] {
                model.stream.set_prefill_materialization(mode);
                model.clear_prefix_cache();
                let mut values = Vec::new();
                for reuse in 0..3 {
                    let requests = (0..width)
                        .map(|row| {
                            let mut tokens = (0..65).map(|i| (i + row) % 64).collect::<Vec<u32>>();
                            if reuse == 2 {
                                tokens[40] = (tokens[40] + 7) % 64;
                            }
                            let request = PrefillRequest {
                                model: ModelHandle {
                                    id: model.info.manifest.id.clone(),
                                    backend: "metal".into(),
                                },
                                session_id: Uuid::new_v4(),
                                prompt_tokens: tokens,
                                cache_checkpoints: vec![32],
                                block_table: BlockTable::with_block_size(16),
                                cached_tokens: 0,
                                generation_tokens: None,
                                sampling_logits: SamplingLogits::Full,
                            };
                            (request, SamplingLogits::Full)
                        })
                        .collect();
                    let (batch, _) = MetalPrefillBatch::prepare(&mut model, requests, None)?;
                    let mut complete = false;
                    for _ in 0..512 {
                        if batch.execute_step(&mut model, budget)?.complete {
                            complete = true;
                            break;
                        }
                    }
                    assert!(complete);
                    for row in batch.finish()? {
                        assert_eq!(row.native.prefix_cache_tokens, [0, 65, 32][reuse]);
                        let NativeOutput::Logits(logits) = row.native.output else {
                            return Err(Error::InvalidPrefillBatch("expected full logits".into()));
                        };
                        values.extend(logits.to_vec_f32(model.stream())?);
                        let mut token = logits.argmax_u32(model.stream())?;
                        for _ in 0..16 {
                            let NativeOutput::Logits(logits) = model.decode(
                                row.request.session_id,
                                token,
                                SamplingLogits::Full,
                            )?
                            else {
                                return Err(Error::InvalidPrefillBatch(
                                    "expected decode logits".into(),
                                ));
                            };
                            values.extend(logits.to_vec_f32(model.stream())?);
                            token = logits.argmax_u32(model.stream())?;
                        }
                        model.release_session(row.request.session_id)?;
                    }
                }
                assert!(values.iter().all(|v| v.is_finite()));
                if let Some(reference) = &reference {
                    assert_eq!(&values, reference);
                } else {
                    reference = Some(values);
                }
            }
        }
        model.clear_prefix_cache();
        assert_eq!(model.stream().paged_arenas().resident_arenas()?, 0);
        drop(model);
        std::fs::remove_dir_all(directory)?;
    }
    Ok(())
}

#[test]
fn state_only_materializes_cache_without_the_unused_final_output() -> Result<()> {
    use crate::{
        engine::probe,
        native::{prefill::evaluation, session::SessionState, step},
    };
    for (family, page_min_context) in
        [(Family::Hybrid, 1), (Family::Clamped, 1), (Family::Hybrid, 128), (Family::Clamped, 128)]
    {
        let (mut model, directory) = fixture::load_family_with_paging(family, 0, page_min_context)?;
        for mode in [OutputAndState, StateOnly] {
            model.stream.set_prefill_materialization(mode);
            model.reserve_prefill_pages(32)?;
            let decoder = model.execution.decoder()?;
            let mut cache = decoder.new_cache(model.stream())?;
            cache.reserve(48)?;
            cache.plan_contiguous(48);
            let mut state = SessionState::new(cache);
            let hidden = step::forward_prefill_state(
                decoder,
                model.stream(),
                &mut state,
                &(0..32).collect::<Vec<_>>(),
                0,
            )?;
            assert!(!probe::is_materialized(&hidden)?);
            evaluation::intermediate(&model, &state, hidden.snapshot()?)?;
            assert_eq!(state.cache.cached_tokens()?, 32);
            assert_eq!(probe::is_materialized(&hidden)?, mode == OutputAndState);
            let mut roots = Vec::new();
            state.cache.extend_prefill_graph_roots(&mut roots);
            if page_min_context == 128 {
                assert!(!roots.is_empty());
            }
            for root in roots {
                assert!(probe::is_materialized(root)?, "persistent cache must be evaluated");
            }
            // Both recurrent and final full-attention writes must survive into decode.
            let logits = step::forward_token(decoder, model.stream(), &mut state, 32, 32, false)?;
            evaluation::materialize(&model, &state, &logits)?;
            assert!(logits.to_vec_f32(model.stream())?.iter().all(|v| v.is_finite()));
        }
        assert_eq!(model.stream().paged_arenas().resident_arenas()?, 0);
        drop(model);
        std::fs::remove_dir_all(directory)?;
    }
    Ok(())
}

#[test]
fn state_only_preserves_final_sliding_layer_after_wrap()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let (model, directory) = load_family(Family::Clamped, 16)?;
    drop(model);
    let config_path = directory.join("config.json");
    let mut config: serde_json::Value = serde_json::from_slice(&std::fs::read(&config_path)?)?;
    config["layer_types"] = serde_json::json!(["sliding_attention"]);
    config["sliding_window"] = serde_json::json!(16);
    std::fs::write(config_path, serde_json::to_vec(&config)?)?;
    let mut model =
        fixture::load_path_with_paging(directory.to_string_lossy().into_owned(), 16, 256, 128)?;
    model.info.prefill_step = 16;
    let mut reference = None;
    for mode in [OutputAndState, StateOnly] {
        model.stream.set_prefill_materialization(mode);
        model.clear_prefix_cache();
        let session = Uuid::new_v4();
        let prompt = (0..65).map(|index| index % 64).collect::<Vec<_>>();
        let output =
            model.prefill(session, &prompt, &[32], &SamplingLogits::Full, Some(16), &mut |_| {})?;
        let NativeOutput::Logits(logits) = output.output else {
            return Err(Error::InvalidPrefillBatch("expected full prefill logits".into()).into());
        };
        let mut values = logits.to_vec_f32(model.stream())?;
        let mut token = logits.argmax_u32(model.stream())?;
        for _ in 0..16 {
            let NativeOutput::Logits(logits) =
                model.decode(session, token, SamplingLogits::Full)?
            else {
                return Err(Error::InvalidPrefillBatch("expected full decode logits".into()).into());
            };
            values.extend(logits.to_vec_f32(model.stream())?);
            token = logits.argmax_u32(model.stream())?;
        }
        assert!(values.iter().all(|value| value.is_finite()));
        if let Some(reference) = &reference {
            assert_eq!(&values, reference);
        }
        reference = Some(values);
        model.release_session(session)?;
    }
    model.clear_prefix_cache();
    assert_eq!(model.stream().paged_arenas().resident_arenas()?, 0);
    drop(model);
    std::fs::remove_dir_all(directory)?;
    Ok(())
}
