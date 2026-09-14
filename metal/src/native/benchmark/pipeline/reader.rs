use std::{io::Write, sync::Arc};

use runtime::{backend::SamplingLogits, tuning::TuningMode};
use uuid::Uuid;

use super::BenchmarkConfig;
use crate::{
    engine::{Array, DecoderCache},
    native::{
        error::Result,
        model::{LoadedModel, NativeOutput},
    },
};

#[test]
#[ignore = "teacher-forced reader diagnosis on a real model; set MIRMIR_BENCH_MODEL"]
fn locates_scalar_and_packed_reader_divergence() -> Result<()> {
    let context = super::super::positive_env("MIRMIR_BENCH_PIPELINE_CONTEXT", 8193)?;
    let config = BenchmarkConfig {
        model: super::super::model_path()?,
        prompt_tokens: context,
        decode_tokens: 128,
        samples: 1,
        warmup: 0,
    };
    let mut settings = super::super::diagnostics::isolated_config();
    let metal = Arc::make_mut(&mut settings);
    metal.tuning.mode = TuningMode::Disabled;
    metal.cache.prefix_cache_entries = 0;
    metal.set_max_batch_requests(6);
    let mut model = LoadedModel::load_with_config(&config.manifest(), settings, &mut |_| {})?;
    let mut tokens = Vec::new();
    let mut caches = [Vec::new(), Vec::new(), Vec::new()];
    for row in 0..2 {
        let prompt = (0..context)
            .map(|index| Ok(u32::try_from((row * 1009 + index) % 100_000 + 1000)?))
            .collect::<Result<Vec<_>>>()?;
        let session = Uuid::new_v4();
        let output =
            model.prefill(session, &prompt, &[], SamplingLogits::Full, None, &mut |_| {})?;
        let NativeOutput::Logits(logits) = output.output else {
            unreachable!()
        };
        tokens.push(logits.argmax_u32(model.stream())?);
        let state = model.sessions.remove(&session).ok_or(super::super::Error::NoPendingDecode)?;
        for cohort in &mut caches {
            cohort.push(state.cache.snapshot_at(context)?);
        }
        writeln!(std::io::stderr().lock(), "reader.prefill: row={row}, context={context}")?;
    }
    let mut first_difference = None;
    for step in 0..128 {
        let mut values = Vec::new();
        for (variant, cohort) in caches.iter_mut().enumerate() {
            model.stream.set_force_native_paged_attention(variant == 1);
            let output = forward(&model, cohort, &tokens, context + step, variant == 2)?;
            let mut roots = output.iter().collect::<Vec<_>>();
            for cache in cohort {
                cache.extend_graph_roots(&mut roots);
            }
            model.stream().eval_many_with_paged_arenas(&roots)?;
            values.push(
                output
                    .iter()
                    .map(|output| output.to_vec_f32(model.stream()))
                    .collect::<crate::engine::Result<Vec<_>>>()?,
            );
        }
        for (row, token) in tokens.iter_mut().enumerate() {
            let ranked = values.iter().map(|variant| rank(&variant[row])).collect::<Vec<_>>();
            let differs = ranked[0][0].0 != ranked[2][0].0;
            let native_differs = ranked[1][0].0 != ranked[2][0].0;
            if step == 0 || differs || native_differs {
                writeln!(
                    std::io::stderr().lock(),
                    "reader.teacher_forced: step={step}, row={row}, auto={:?}, native={:?}, packed={:?}, auto_packed_max_abs={}, native_packed_max_abs={}",
                    ranked[0],
                    ranked[1],
                    ranked[2],
                    max_error(&values[0][row], &values[2][row]),
                    max_error(&values[1][row], &values[2][row])
                )?;
            }
            assert!(!native_differs, "reader-aligned scalar/batch differs at {step}/{row}");
            if differs {
                first_difference.get_or_insert((step, row));
            }
            *token = u32::try_from(ranked[0][0].0)?;
        }
        if first_difference.is_some() {
            break;
        }
    }
    model.stream().synchronize()?;
    writeln!(std::io::stderr().lock(), "reader.first_difference: {first_difference:?}")?;
    Ok(())
}

fn forward(
    model: &LoadedModel,
    caches: &mut [DecoderCache],
    tokens: &[u32],
    position: usize,
    packed: bool,
) -> Result<Vec<Array>> {
    let stream = model.stream();
    let decoder = model.execution.decoder()?;
    let position = i32::try_from(position)?;
    if packed {
        let output = decoder.forward_packed_decode(
            &Array::from_u32(tokens, &[i32::try_from(tokens.len())?, 1])?,
            &mut caches.iter_mut().collect::<Vec<_>>(),
            &vec![position; tokens.len()],
            stream,
        )?;
        let width = usize::try_from(output.shape()?[2])?;
        return (0..tokens.len())
            .map(|row| Ok(output.slice(&[row, 0, 0], &[row + 1, 1, width], stream)?))
            .collect();
    }
    tokens
        .iter()
        .zip(caches)
        .map(|(token, cache)| {
            Ok(decoder.forward_decode(
                &Array::from_u32(&[*token], &[1, 1])?,
                cache,
                position,
                stream,
            )?)
        })
        .collect()
}

fn rank(logits: &[f32]) -> Vec<(usize, f32)> {
    assert!(logits.iter().all(|value| value.is_finite()));
    let mut sorted = logits.iter().copied().enumerate().collect::<Vec<_>>();
    sorted.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    sorted.truncate(5);
    sorted
}

fn max_error(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max)
}
