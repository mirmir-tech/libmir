use super::*;

pub(super) fn run_width_with_decode(
    model: &mut LoadedModel,
    mode: MoePrefill,
    context: usize,
    width: usize,
    decode: impl FnOnce(&mut LoadedModel, &mut [DecodeInput]) -> Result<decode::Observation>,
) -> Result<Observation> {
    run_budget_with_decode(model, mode, context, width, None, decode)
}

pub(super) fn run_budget_with_decode(
    model: &mut LoadedModel,
    mode: MoePrefill,
    context: usize,
    width: usize,
    generation_tokens: Option<std::num::NonZeroUsize>,
    decode: impl FnOnce(&mut LoadedModel, &mut [DecodeInput]) -> Result<decode::Observation>,
) -> Result<Observation> {
    assert!((2..=5).contains(&width));
    assert!(model.sessions.is_empty());
    model.stream.synchronize()?;
    model.stream.set_moe_prefill(mode);
    model.clear_prefix_cache();
    let requests = (0..width)
        .map(|row| {
            let (mut request, sampling) = request(model, 0, row, context)?;
            request.generation_tokens = generation_tokens;
            Ok((request, sampling))
        })
        .collect::<Result<Vec<_>>>()?;
    let started = Instant::now();
    let (batch, _) = MetalPrefillBatch::prepare(model, requests, None)?;
    let mut schedule = Vec::new();
    let mut complete = false;
    for iteration in 0..64 {
        let step = batch.execute_step(model, width * 512)?;
        let positions = step
            .events
            .iter()
            .map(|(row, event)| (*row, event.count().current()))
            .collect::<Vec<_>>();
        if !step.complete {
            assert_eq!(positions.len(), width, "cohort changed under memory pressure");
            assert!(
                positions
                    .iter()
                    .all(|&(row, position)| row < width && position == (iteration + 1) * 512)
            );
        }
        schedule.push(positions);
        if step.complete {
            complete = true;
            break;
        }
    }
    assert!(complete);
    let mut inputs = batch
        .finish()?
        .into_iter()
        .map(|finished| {
            Ok(DecodeInput {
                session: finished.request.session_id,
                token: greedy_token(&finished.native.output)?,
                sampling: SamplingLogits::None,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    model.stream.synchronize()?;
    let prefill_ms = started.elapsed().as_secs_f64() * 1000.0;
    let memory = crate::engine::memory_stats()?;
    let decode::Observation { tokens, elapsed_ms: decode_ms } = decode(model, &mut inputs)?;
    for input in inputs {
        model.release_session(input.session)?;
    }
    Ok(Observation {
        prefill_ms,
        decode_ms,
        tokens,
        schedule,
        active_bytes: memory.active,
        cached_bytes: memory.cached,
        peak_bytes_process: memory.peak,
    })
}
