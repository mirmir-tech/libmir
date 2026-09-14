mod metrics;

use super::*;
use crate::engine::{Array, DecoderCache, probe};

#[test]
#[ignore = "matched-cache scalar/packed layer capture on real weights"]
fn localizes_decode_width_difference() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut model = fixture::load_path(std::env::var("MIRMIR_BENCH_MODEL")?, 0)?;
    let rows = wave(&mut model, &[0, 1, 2, 3, 4], false)?;
    model.flush_decode_graphs()?;
    let mut caches = Vec::new();
    let mut tokens = Vec::new();
    for input in &rows {
        let state = &model.sessions[&input.session];
        let pending = state.pending.as_ref().ok_or(Error::NoPendingDecode)?;
        tokens.push(pending.logits.argmax_u32(model.stream())?);
        caches.push(state.cache.snapshot_at(state.cache.cached_tokens()?)?);
    }
    for input in rows {
        model.release_session(input.session)?;
    }
    for step in 0..9 {
        let capture = matches!(step, 0 | 8);
        let mut packed_caches = snapshots(&caches)?;
        let mut control_caches = snapshots(&caches)?;
        let scalar = forward(&model, &mut caches, &tokens, false, capture)?;
        if capture {
            let packed = forward(&model, &mut packed_caches, &tokens, true, true)?;
            let mut packed_control = snapshots(&control_caches)?;
            let control = forward(&model, &mut control_caches, &tokens, false, false)?;
            metrics::assert_identical(&scalar, &control, "scalar capture perturbs logits");
            let control = forward(&model, &mut packed_control, &tokens, true, false)?;
            metrics::assert_identical(&packed, &control, "packed capture perturbs logits");
            metrics::report(step, &scalar, &packed)?;
            for projection in &packed.projections {
                projection.report(step, model.stream())?;
            }
        }
        tokens = scalar
            .logits
            .iter()
            .map(|row| metrics::argmax(row))
            .collect::<Result<Vec<_>>>()?;
    }
    drop(caches);
    model.clear_prefix_cache();
    assert_eq!(model.stream().paged_arenas().resident_arenas()?, 0);
    Ok(())
}

fn snapshots(caches: &[DecoderCache]) -> Result<Vec<DecoderCache>> {
    caches
        .iter()
        .map(|cache| Ok(cache.snapshot_at(cache.cached_tokens()?)?))
        .collect()
}

fn forward(
    model: &LoadedModel,
    caches: &mut [DecoderCache],
    tokens: &[u32],
    packed: bool,
    capture: bool,
) -> Result<Frame> {
    let stream = model.stream();
    let decoder = model.execution.decoder()?;
    let guard = capture
        .then(|| {
            probe::Capture::begin(&[
                (19, probe::ProjectionKind::QueryGate),
                (7, probe::ProjectionKind::Output),
            ])
        })
        .transpose()?;
    let mut outputs = Vec::new();
    let positions = caches
        .iter()
        .map(|cache| Ok(i32::try_from(cache.cached_tokens()?)?))
        .collect::<Result<Vec<_>>>()?;
    if packed {
        outputs.push(decoder.forward_packed_decode(
            &Array::from_u32(tokens, &[i32::try_from(tokens.len())?, 1])?,
            &mut caches.iter_mut().collect::<Vec<_>>(),
            &positions,
            stream,
        )?);
    } else {
        for ((cache, token), position) in caches.iter_mut().zip(tokens).zip(positions) {
            outputs.push(decoder.forward_decode(
                &Array::from_u32(&[*token], &[1, 1])?,
                cache,
                position,
                stream,
            )?);
        }
    }
    let (stages, projections) = guard
        .map(probe::Capture::finish)
        .transpose()?
        .map(|trace| (trace.values, trace.projections))
        .unwrap_or_default();
    let mut roots = outputs.iter().collect::<Vec<_>>();
    roots.extend(stages.iter().map(|value| &value.array));
    roots.extend(projections.iter().map(|value| &value.input));
    for cache in caches {
        cache.extend_graph_roots(&mut roots);
    }
    stream.eval_many_with_paged_arenas(&roots)?;
    stream.synchronize()?;
    let mut values = Vec::new();
    for output in outputs {
        let width = usize::try_from(output.shape()?[2])?;
        values.extend(output.to_vec_f32(stream)?.chunks_exact(width).map(<[f32]>::to_vec));
    }
    let stages = stages
        .into_iter()
        .map(|v| {
            Ok(HostStage {
                layer: v.layer,
                stage: v.stage,
                values: v.array.to_vec_f32(stream)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Frame { logits: values, stages, projections })
}

struct Frame {
    logits: Vec<Vec<f32>>,
    stages: Vec<HostStage>,
    projections: Vec<probe::Projection>,
}

struct HostStage {
    layer: usize,
    stage: probe::Stage,
    values: Vec<f32>,
}
