use super::*;
use crate::native::model::NativeOutput;

struct Prediction {
    token: u32,
    logits: Vec<f32>,
}

#[test]
#[ignore = "teacher-forced terminal-prefill logits; set MIRMIR_BENCH_REFERENCE_TOKENS"]
fn compares_logits_on_identical_history() -> Result<()> {
    let config = BenchmarkConfig::from_env()?;
    let reference_path = std::env::var("MIRMIR_BENCH_REFERENCE_TOKENS")
        .map_err(|error| crate::native::error::Error::Benchmark(error.to_string()))?;
    let history: Vec<u32> =
        serde_json::from_slice(&std::fs::read(reference_path)?).map_err(|error| {
            crate::native::error::Error::Benchmark(format!("invalid reference tokens: {error}"))
        })?;
    assert_eq!(history.len(), 32);
    let mut settings = diagnostics::isolated_config();
    let metal = Arc::make_mut(&mut settings);
    metal.tuning.mode = TuningMode::Disabled;
    metal.cache.prefix_cache_entries = 4;
    let mut model = LoadedModel::load_with_config(&config.manifest(), settings, &mut |_| {})?;
    model.info.prefill_step = 512;
    let prompt = (1_000..3_065).collect::<Vec<_>>();
    model.stream.set_prefill_output(SeparateToken);
    let baseline = capture(&mut model, &prompt, &history)?;
    model.stream.set_prefill_output(LastChunk);
    let candidate = capture(&mut model, &prompt, &history)?;
    for (step, (a, b)) in baseline.iter().zip(&candidate).enumerate() {
        assert_eq!(a.logits.len(), b.logits.len());
        assert_eq!(a.token, history[step], "full logits changed the original baseline");
        let differences = a.logits.iter().zip(&b.logits).map(|(a, b)| f64::from(a - b));
        let maximum = differences.clone().map(f64::abs).fold(0.0, f64::max);
        let rmse = (differences.map(|difference| difference * difference).sum::<f64>()
            / f64::from(u32::try_from(a.logits.len())?))
        .sqrt();
        writeln!(
            std::io::stderr().lock(),
            "terminal.logits: {}",
            json!({"step": step, "baseline_token": a.token, "candidate_token": b.token,
                "max_abs_difference": maximum, "rmse": rmse,
                "baseline_top_logit": a.logits[a.token as usize],
                "candidate_top_logit": b.logits[b.token as usize],
                "baseline_margin": margin(a), "candidate_margin": margin(b),
                "candidate_baseline_token_logit": b.logits[a.token as usize],
                "baseline_candidate_token_logit": a.logits[b.token as usize],
                "history_source": "original separate-token greedy trajectory"}),
        )?;
    }
    model.clear_prefix_cache();
    assert_eq!(model.stream().paged_arenas().resident_arenas()?, 0);
    Ok(())
}

fn capture(model: &mut LoadedModel, prompt: &[u32], history: &[u32]) -> Result<Vec<Prediction>> {
    model.clear_prefix_cache();
    let session = Uuid::new_v4();
    let output =
        model.prefill(session, prompt, &[2048], &SamplingLogits::Full, Some(16), &mut |_| {})?;
    let mut predictions = vec![prediction(model, output.output)?];
    for &token in &history[..history.len() - 1] {
        let output = model.decode(session, token, SamplingLogits::Full)?;
        predictions.push(prediction(model, output)?);
    }
    model.release_session(session)?;
    Ok(predictions)
}

fn prediction(model: &LoadedModel, output: NativeOutput) -> Result<Prediction> {
    let NativeOutput::Logits(logits) = output else {
        return Err(crate::native::error::Error::Benchmark("expected full logits".into()));
    };
    let token = logits.argmax_u32(model.stream())?;
    let logits = logits.to_vec_f32(model.stream())?;
    assert!(logits.iter().all(|value| value.is_finite()));
    Ok(Prediction { token, logits })
}

fn margin(prediction: &Prediction) -> f32 {
    let second = prediction
        .logits
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != prediction.token as usize)
        .map(|(_, value)| *value)
        .fold(f32::NEG_INFINITY, f32::max);
    prediction.logits[prediction.token as usize] - second
}
