mod semantic;

use std::{io::Write, sync::Arc};

use runtime::{backend::SamplingLogits, tuning::TuningMode};
use uuid::Uuid;

use super::{BenchmarkConfig, output_token};
use crate::native::{
    error::Result,
    model::{DecodeExecution, DecodeInput, LoadedModel},
};

const STEPS: usize = 64;

#[test]
#[ignore = "real-model long shared prefix, mixed sampling, branch isolation and refill"]
fn preserves_shared_prefix_branches_through_mixed_decode() -> Result<()> {
    let context = super::super::positive_env("MIRMIR_BENCH_PIPELINE_CONTEXT", 8193)?;
    let config = BenchmarkConfig {
        model: super::super::model_path()?,
        prompt_tokens: context + 4,
        decode_tokens: STEPS + 1,
        samples: 1,
        warmup: 0,
    };
    let mut settings = super::super::diagnostics::isolated_config();
    let metal = Arc::make_mut(&mut settings);
    metal.tuning.mode = TuningMode::Disabled;
    metal.cache.prefix_cache_entries = 8;
    metal.set_max_batch_requests(4);
    let blocks = (context + STEPS + 5).div_ceil(metal.kv_cache.block_size) * 4;
    metal.kv_cache.block_count = metal.kv_cache.block_count.max(u32::try_from(blocks)?);
    let mut model = LoadedModel::load_with_config(&config.manifest(), settings, &mut |_| {})?;
    let prompt = (0..context)
        .map(|index| Ok(u32::try_from(index % 100_000 + 1000)?))
        .collect::<Result<Vec<_>>>()?;
    let (base, hit) = prefill(&mut model, &prompt, SamplingLogits::Full)?;
    assert_eq!(hit, 0);
    let expected_base = reference(&mut model, base)?;
    let mut extended = prompt.clone();
    extended.extend([1234, 4321, 5678]);
    let (branch, hit) = prefill(&mut model, &extended, SamplingLogits::Full)?;
    assert!(hit > 0 && hit <= context, "continuation did not reuse the base prefix");
    let expected_branch = reference(&mut model, branch)?;
    let mut rows = Vec::new();
    for (tokens, expected) in
        [(&prompt, &expected_base), (&prompt, &expected_base), (&extended, &expected_branch)]
    {
        let (input, hit) = prefill(&mut model, tokens, SamplingLogits::None)?;
        assert_eq!(hit, tokens.len(), "terminal prefix was not reused exactly");
        assert_eq!(input.token, expected[0]);
        rows.push((input, expected, 0));
    }
    // A diagnostic control keeps cancelled rows resident but never schedules them.
    let retain_inactive = std::env::var_os("MIRMIR_BENCH_PREFIX_RETAIN_INACTIVE").is_some();
    let mut inactive = Vec::new();
    let mut packed = 0;
    let mut checked = 0;
    let mut digest = blake3::Hasher::new();
    for step in 0..STEPS {
        if step == 16 {
            let (cancelled, _, _) = rows.remove(1);
            if retain_inactive {
                inactive.push(cancelled);
            } else {
                model.release_session(cancelled.session)?;
            }
        }
        if step == 32 {
            let (refill, hit) = prefill(&mut model, &prompt, SamplingLogits::None)?;
            assert_eq!(hit, context);
            assert_eq!(refill.token, expected_base[0]);
            rows.push((refill, &expected_base, 0));
            rows.rotate_left(1);
        }
        for (row, (input, _, _)) in rows.iter_mut().enumerate() {
            input.sampling = if step % 16 < 4 && row == 0 {
                SamplingLogits::Full
            } else {
                SamplingLogits::None
            };
        }
        let inputs = rows.iter().map(|(input, _, _)| *input).collect::<Vec<_>>();
        let outputs = model.decode_grouped(&inputs)?;
        assert_eq!(outputs.len(), rows.len());
        for (row, ((input, expected, offset), (output, execution))) in
            rows.iter_mut().zip(outputs).enumerate()
        {
            *offset += 1;
            input.token = output_token(&model, output)?;
            assert_eq!(
                input.token, expected[*offset],
                "context {context}, step {step}, row {row}, offset {offset}, retain_inactive {retain_inactive}"
            );
            packed += usize::from(matches!(execution, DecodeExecution::Packed { .. }));
            checked += 1;
            digest.update(&input.token.to_le_bytes());
        }
    }
    assert!(packed > STEPS, "mixed cohort did not execute packed decode");
    for (input, _, _) in rows {
        model.release_session(input.session)?;
    }
    for input in inactive {
        model.release_session(input.session)?;
    }
    assert!(model.sessions.is_empty());
    model.clear_prefix_cache();
    assert_eq!(model.stream().paged_arenas().resident_arenas()?, 0);
    writeln!(
        std::io::stderr().lock(),
        "prefix.mixed: context={context}, checked={checked}, packed={packed}, digest={}, all arenas released",
        digest.finalize()
    )?;
    Ok(())
}

fn prefill(
    model: &mut LoadedModel,
    prompt: &[u32],
    sampling: SamplingLogits,
) -> Result<(DecodeInput, usize)> {
    let session = Uuid::new_v4();
    let output = model.prefill(session, prompt, &[], sampling, None, &mut |_| {})?;
    let token = output_token(model, output.output)?;
    Ok((DecodeInput { session, token, sampling }, output.prefix_cache_tokens))
}

fn reference(model: &mut LoadedModel, mut input: DecodeInput) -> Result<Vec<u32>> {
    let mut tokens = vec![input.token];
    for _ in 0..STEPS {
        let output = model.decode(input.session, input.token, SamplingLogits::Full)?;
        input.token = output_token(model, output)?;
        tokens.push(input.token);
    }
    model.release_session(input.session)?;
    Ok(tokens)
}
