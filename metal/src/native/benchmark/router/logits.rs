use std::io::Write;

use runtime::backend::SamplingLogits;
use uuid::Uuid;

use super::*;
use crate::native::{error::Error, model::NativeOutput};

#[test]
#[ignore = "real model full-vocabulary first-token logits and target probabilities"]
fn compares_router_precision_logits() -> Result<()> {
    let mut config = super::super::BenchmarkConfig::from_env()?;
    config.prompt_tokens = 512;
    let tokenizer = TextTokenizer::from_layout(&ModelLayout::inspect(&config.model)?)?;
    let prompts = prompts(&tokenizer)?;
    let mut settings = super::super::diagnostics::isolated_config();
    Arc::make_mut(&mut settings).tuning.mode = runtime::tuning::TuningMode::Disabled;
    let mut model = LoadedModel::load_with_config(&config.manifest(), settings, &mut |_| {})?;
    for (index, (prompt, (_, answer))) in prompts.iter().zip(CASES).enumerate() {
        let ids = tokenizer.encode_with_special_tokens(answer, false)?.token_ids;
        let target =
            usize::try_from(*ids.first().ok_or_else(|| Error::Benchmark("empty answer".into()))?)?;
        let a = logits(&mut model, prompt, RouterPrecision::Model)?;
        let b = logits(&mut model, prompt, RouterPrecision::Float32)?;
        assert_eq!(a.len(), b.len());
        assert!(a.iter().chain(&b).all(|v| v.is_finite()));
        let a_logz = logz(&a);
        let b_logz = logz(&b);
        let kl = a
            .iter()
            .zip(&b)
            .map(|(a, b)| {
                let logp = f64::from(*a) - a_logz;
                logp.exp() * (logp - (f64::from(*b) - b_logz))
            })
            .sum::<f64>();
        let max_abs = a.iter().zip(&b).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        let argmax = |values: &[f32]| super::super::argmax(values).unwrap_or(0);
        writeln!(
            std::io::stderr().lock(),
            "router.logits: {}",
            serde_json::json!({
            "case":index,"max_abs":max_abs,"kl_model_to_fp32":kl,
            "argmax_model":argmax(&a),"argmax_fp32":argmax(&b),"target":target,
            "target_nll_model":a_logz-f64::from(a[target]),"target_nll_fp32":b_logz-f64::from(b[target]) })
        )?;
    }
    Ok(())
}

fn logits(model: &mut LoadedModel, prompt: &[u32], precision: RouterPrecision) -> Result<Vec<f32>> {
    model.stream.synchronize()?;
    model.stream.set_router_precision_for_test(precision);
    let session = Uuid::new_v4();
    let output = model.prefill(session, prompt, &[], SamplingLogits::Full, None, &mut |_| {})?;
    let NativeOutput::Logits(logits) = output.output else {
        return Err(Error::Benchmark("missing logits".into()));
    };
    let values = logits.to_vec_f32(model.stream())?;
    model.release_session(session)?;
    Ok(values)
}

fn logz(values: &[f32]) -> f64 {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    f64::from(max) + values.iter().map(|v| (f64::from(*v) - f64::from(max)).exp()).sum::<f64>().ln()
}
