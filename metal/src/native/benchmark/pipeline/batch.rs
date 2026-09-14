use runtime::backend::SamplingLogits;
use uuid::Uuid;

use super::{BenchmarkConfig, CONTEXT_TOKENS, DECODE_TOKENS, generate, output_token};
use crate::native::{
    error::Result,
    model::{DecodeInput, LoadedModel},
};

#[test]
#[ignore = "loads a real model; set MIRMIR_BENCH_MODEL or MODEL"]
fn preserves_batched_pipeline_against_synchronous_logits() -> Result<()> {
    let context = super::super::positive_env("MIRMIR_BENCH_PIPELINE_CONTEXT", CONTEXT_TOKENS)?;
    let config = BenchmarkConfig {
        model: super::super::model_path()?,
        decode_tokens: DECODE_TOKENS,
        prompt_tokens: context,
        samples: 1,
        warmup: 0,
    };
    let mut ignored = |_event| {};
    let mut model = LoadedModel::load_with_config(
        &config.manifest(),
        super::super::diagnostics::isolated_config(),
        &mut ignored,
    )?;
    let prompts = (0..2)
        .map(|seed| {
            (0..context)
                .map(|index| Ok(u32::try_from((seed * 1_009 + index) % 100_000 + 1_000)?))
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()?;
    let scalar = prompts
        .iter()
        .map(|prompt| generate(&mut model, prompt, SamplingLogits::Full))
        .collect::<Result<Vec<_>>>()?;
    let expected = synchronous_batch(&mut model, &prompts)?;
    let mut inputs = Vec::new();
    for prompt in &prompts {
        let session = Uuid::new_v4();
        let output =
            model.prefill(session, prompt, &[], SamplingLogits::None, None, &mut ignored)?;
        inputs.push(DecodeInput {
            session,
            token: output_token(&model, output.output)?,
            sampling: SamplingLogits::None,
        });
    }
    for (step, expected) in expected.iter().enumerate() {
        for (row, input) in inputs.iter().enumerate() {
            assert_eq!(input.token, expected[row], "batch row {row}, step {step}");
            assert_eq!(input.token, scalar[row][step], "single/batch row {row}, step {step}");
        }
        if step == DECODE_TOKENS {
            break;
        }
        let outputs = model.decode_batch(&inputs)?;
        for (input, output) in inputs.iter_mut().zip(outputs) {
            input.token = output_token(&model, output)?;
        }
    }
    for input in inputs {
        model.release_session(input.session)?;
    }
    Ok(())
}

// Keep the same packed math, but finish sampling on the host before
// constructing the next forward graph. Scalar and packed BF16 kernels need not
// round alike.
fn synchronous_batch(model: &mut LoadedModel, prompts: &[Vec<u32>]) -> Result<Vec<Vec<u32>>> {
    use crate::{
        engine::Array,
        native::{session::PendingDecode, step},
    };

    let mut states = Vec::new();
    let mut tokens = Vec::new();
    for prompt in prompts {
        let session = Uuid::new_v4();
        let output =
            model.prefill(session, prompt, &[], SamplingLogits::None, None, &mut |_event| {})?;
        tokens.push(output_token(model, output.output)?);
        states.push(model.sessions.remove(&session).ok_or_else(|| {
            crate::native::error::Error::Benchmark("prefilled session is missing".into())
        })?);
    }
    let mut expected = vec![tokens.clone()];
    let stream = model.stream();
    for _ in 0..DECODE_TOKENS {
        for (state, token) in states.iter_mut().zip(&mut tokens) {
            let logits = step::take_pending(state, *token)?;
            *token = logits.argmax_u32(stream)?;
        }
        let token_ids = Array::from_u32(&tokens, &[i32::try_from(tokens.len())?, 1])?;
        let positions = states
            .iter()
            .map(|state| Ok(i32::try_from(state.model_position()?)?))
            .collect::<Result<Vec<_>>>()?;
        let mut caches = states.iter_mut().map(|state| &mut state.cache).collect::<Vec<_>>();
        let logits = model
            .execution
            .decoder()?
            .forward_packed_decode(&token_ids, &mut caches, &positions, stream)?;
        let mut roots = vec![&logits];
        for state in &states {
            state.cache.extend_graph_roots(&mut roots);
        }
        stream.eval_many_with_paged_arenas(&roots)?;
        let width = usize::try_from(logits.shape()?[2])?;
        for (row, (state, token_id)) in states.iter_mut().zip(&tokens).enumerate() {
            state.pending = Some(PendingDecode {
                token_id: *token_id,
                logits: logits.slice(&[row, 0, 0], &[row + 1, 1, width], stream)?,
            });
        }
        expected.push(tokens.clone());
    }
    stream.synchronize()?;
    Ok(expected)
}
