use super::*;
use crate::{
    engine::probe,
    native::{prefill::evaluation, session::SessionState, step},
};

struct Captured {
    values: Vec<(usize, probe::Stage, Vec<f32>)>,
    routes: Vec<(usize, Vec<u32>)>,
}

#[derive(Clone, Copy, serde::Serialize)]
#[serde(rename_all = "snake_case")]
enum TailStep {
    Prefill,
    Decode,
}

#[test]
#[ignore = "localize terminal-prefill layer changes from identical checkpoint state"]
fn localizes_terminal_prefill_differences() -> Result<()> {
    let config = BenchmarkConfig::from_env()?;
    let settings = configured_settings()?;
    let mut model = LoadedModel::load_with_config(&config.manifest(), settings, &mut |_| {})?;
    let prompt = (1_000..3_065).collect::<Vec<_>>();
    model.reserve_prefill_pages(400)?;
    let decoder = model.execution.decoder()?;
    let mut cache = decoder.new_cache(model.stream())?;
    cache.reserve(2_081)?;
    cache.plan_contiguous(2_081);
    let mut common = SessionState::new(cache);
    for position in (0..2048).step_by(512) {
        let hidden = step::forward_prefill_state(
            decoder,
            model.stream(),
            &mut common,
            &prompt[position..position + 512],
            position,
        )?;
        evaluation::materialize(&model, &common, &hidden)?;
    }
    common.position = 2048;
    assert_eq!(common.cache.cached_tokens()?, 2048);
    let mut scalar = common.snapshot()?;
    let mut chunk = common.snapshot()?;
    let prefix = capture(&model, &mut scalar, &prompt[2048..2064], 2048, TailStep::Prefill)?;
    let last = capture(&model, &mut scalar, &prompt[2064..], 2064, TailStep::Decode)?;
    let bulk = capture(&model, &mut chunk, &prompt[2048..], 2048, TailStep::Prefill)?;
    assert_eq!(scalar.cache.cached_tokens()?, 2065);
    assert_eq!(chunk.cache.cached_tokens()?, 2065);
    assert_eq!(prefix.values.len(), bulk.values.len());
    assert_eq!(last.values.len(), bulk.values.len());
    for ((a, b), c) in prefix.values.iter().zip(&last.values).zip(&bulk.values) {
        assert_eq!((a.0, a.1), (c.0, c.1));
        assert_eq!((b.0, b.1), (c.0, c.1));
        assert_eq!(a.2.len(), b.2.len() * 16);
        assert_eq!(c.2.len(), b.2.len() * 17);
        writeln!(
            std::io::stderr().lock(),
            "terminal.layer: {}",
            json!({
                "layer": a.0, "stage": a.1,
                "prefix": difference(&a.2, &c.2[..a.2.len()])?,
                "last": difference(&b.2, &c.2[a.2.len()..])?,
            })
        )?;
    }
    for ((a, b), c) in prefix.routes.iter().zip(&last.routes).zip(&bulk.routes) {
        assert_eq!(a.0, c.0);
        assert_eq!(b.0, c.0);
        let width = b.1.len();
        assert_eq!(a.1.len(), 16 * width);
        assert_eq!(c.1.len(), 17 * width);
        let joined = a.1.iter().chain(&b.1).copied().collect::<Vec<_>>();
        let changed = joined
            .chunks(width)
            .zip(c.1.chunks(width))
            .map(|(a, b)| {
                let mut a = a.to_vec();
                let mut b = b.to_vec();
                a.sort_unstable();
                b.sort_unstable();
                a != b
            })
            .collect::<Vec<_>>();
        writeln!(
            std::io::stderr().lock(),
            "terminal.routes: {}",
            json!({
                "layer": a.0, "changed_rows": changed,
            })
        )?;
    }
    drop((common, scalar, chunk));
    assert_eq!(model.stream().paged_arenas().resident_arenas()?, 0);
    Ok(())
}

fn capture(
    model: &LoadedModel,
    state: &mut SessionState,
    tokens: &[u32],
    position: usize,
    step: TailStep,
) -> Result<Captured> {
    let probe = probe::Capture::begin(&[(0, probe::ProjectionKind::GatedDeltaQkv)])?;
    let routes = probe::routing::Capture::begin()?;
    let decoder = model.execution.decoder()?;
    let hidden = if matches!(step, TailStep::Decode) {
        step::forward_token(decoder, model.stream(), state, tokens[0], position, false)?
    } else {
        step::forward_prefill_state(decoder, model.stream(), state, tokens, position)?
    };
    evaluation::materialize(model, state, &hidden)?;
    if position + tokens.len() == 2065 {
        let logits = if matches!(step, TailStep::Decode) {
            hidden
        } else {
            decoder.prefill_logits(&hidden, model.stream())?
        };
        writeln!(
            std::io::stderr().lock(),
            "terminal.first: {}",
            json!({
                "step": step, "token": logits.argmax_u32(model.stream())?,
            })
        )?;
    }
    let trace = probe.finish()?;
    let routes = routes.finish()?;
    if position == 2048 && tokens.len() == 17 {
        assert_eq!(trace.projections.len(), 1);
        for mut projection in trace.projections {
            let width = projection.input.shape()?[2];
            projection.input = projection.input.reshape(&[17, 1, width], model.stream())?;
            projection.report(position, model.stream())?;
        }
    }
    let values = trace
        .values
        .into_iter()
        .filter(|value| {
            matches!(
                value.stage,
                probe::Stage::Input
                    | probe::Stage::Mixer
                    | probe::Stage::FeedForward
                    | probe::Stage::GatedDeltaInput
                    | probe::Stage::GatedDeltaProjection
                    | probe::Stage::GatedDeltaConvolution
                    | probe::Stage::GatedDeltaRecurrence
                    | probe::Stage::GatedDeltaOutput
            )
        })
        .map(|value| Ok((value.layer, value.stage, value.array.to_vec_f32(model.stream())?)))
        .collect::<Result<_>>()?;
    let routes = routes
        .into_iter()
        .map(|value| Ok((value.layer, value.indices.to_vec_u32(model.stream())?)))
        .collect::<Result<_>>()?;
    Ok(Captured { values, routes })
}

fn difference(a: &[f32], b: &[f32]) -> Result<serde_json::Value> {
    assert_eq!(a.len(), b.len());
    assert!(a.iter().chain(b).all(|value| value.is_finite()));
    let delta = a.iter().zip(b).map(|(a, b)| f64::from(a - b));
    let max_abs = delta.clone().map(f64::abs).fold(0.0, f64::max);
    let count = u32::try_from(a.len())?;
    let rmse = (delta.map(|d| d * d).sum::<f64>() / f64::from(count)).sqrt();
    Ok(json!({"max_abs": max_abs, "rmse": rmse}))
}

fn configured_settings() -> Result<Arc<crate::MetalConfig>> {
    let mut settings = diagnostics::isolated_config();
    let metal = Arc::make_mut(&mut settings);
    metal.tuning.mode = TuningMode::Disabled;
    let control = std::env::var("MIRMIR_BENCH_GDN_CONTROL").unwrap_or_default();
    match control.as_str() {
        "" => {},
        "uncompiled" => {
            metal.fusion.compiled_gated_delta_decode = crate::FeatureToggle::Disabled;
        },
        "unfused" => {
            metal.fusion.compiled_gated_delta_decode = crate::FeatureToggle::Disabled;
            metal.fusion.fused_gated_delta_decode = crate::FeatureToggle::Disabled;
        },
        _ => return Err(crate::native::error::Error::Benchmark("invalid GDN control".into())),
    }
    Ok(settings)
}
