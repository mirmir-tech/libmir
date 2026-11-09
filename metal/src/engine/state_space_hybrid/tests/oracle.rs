//! Records greedy continuations of real-checkpoint prompts for comparison
//! with the MLX-LM reference implementation.

use std::{env, fs, path::Path};

use models::{
    execution::DecoderExecutionContract,
    layout::{DecoderConfig, ModelLayout},
    weights::TensorCatalog,
};
use serde_json::{Value, json};

use super::StateSpaceHybridModel;
use crate::engine::{Array, DecoderCache, ModelTensors, Stream, lowering};

type TestResult<T> = Result<T, Box<dyn std::error::Error>>;

#[test]
#[ignore = "needs MIRMIR_BENCH_MODEL, MIRMIR_ORACLE_PROMPTS and MIRMIR_ORACLE_OUTPUT"]
fn records_greedy_tokens_for_the_mlx_lm_oracle() -> TestResult<()> {
    let model_path = env::var("MIRMIR_BENCH_MODEL")?;
    let prompts: Vec<Vec<u32>> =
        serde_json::from_slice(&fs::read(env::var("MIRMIR_ORACLE_PROMPTS")?)?)?;
    let tokens = env::var("MIRMIR_ORACLE_TOKENS").map_or(Ok(48), |value| value.parse())?;
    let root = Path::new(&model_path);
    let layout = ModelLayout::inspect(root)?;
    let decoder = DecoderConfig::from_layout(&layout)?;
    let catalog = TensorCatalog::from_layout(&layout)?;
    let contract = DecoderExecutionContract::discover(&layout, &decoder, &catalog)?;
    let lowering = lowering::plan(&contract.semantic)?;
    let load_stream = Stream::new_cpu()?;
    let tensors = ModelTensors::load(root, &load_stream)?;
    let stream = Stream::new_gpu()?;
    let model = StateSpaceHybridModel::load(
        &tensors,
        &decoder,
        &contract.bindings,
        lowering.layers(),
        256,
        &stream,
    )?;
    let mut records = Vec::new();
    for prompt in prompts {
        records.push(record(&model, &prompt, tokens, &stream)?);
    }
    fs::write(env::var("MIRMIR_ORACLE_OUTPUT")?, serde_json::to_vec(&records)?)?;
    Ok(())
}

fn record(
    model: &StateSpaceHybridModel,
    prompt: &[u32],
    tokens: usize,
    stream: &Stream,
) -> TestResult<Value> {
    let mut cache = model.new_cache(stream)?;
    let (head, tail) = prompt.split_at(prompt.len() - 1);
    let hidden = model.forward_prefill(
        &Array::from_u32(head, &[1, i32::try_from(head.len())?])?,
        &mut cache,
        0,
        stream,
    )?;
    materialize(&hidden, &cache, stream)?;
    let mut generated = Vec::with_capacity(tokens);
    let mut margins = Vec::with_capacity(tokens);
    let mut next = tail[0];
    for step in 0..tokens {
        let logits = model.forward_decode(
            &Array::from_u32(&[next], &[1, 1])?,
            &mut cache,
            i32::try_from(prompt.len() - 1 + step)?,
            stream,
        )?;
        materialize(&logits, &cache, stream)?;
        let values = logits.to_vec_f32(stream)?;
        let (token, margin, top) = greedy(&values);
        generated.push(token);
        margins.push(margin);
        next = token;
        if step == 0 {
            tracing::debug!(top, "first decode logits");
        }
    }
    Ok(json!({ "generated": generated, "margins": margins }))
}

fn materialize(array: &Array, cache: &DecoderCache, stream: &Stream) -> TestResult<()> {
    let mut roots = vec![array];
    cache.extend_prefill_graph_roots(&mut roots);
    stream.eval_many_with_paged_arenas(&roots)?;
    stream.synchronize()?;
    cache.detach_evaluated_graphs(stream)?;
    Ok(())
}

fn greedy(values: &[f32]) -> (u32, f32, f32) {
    let (mut top, mut second, mut index) = (f32::NEG_INFINITY, f32::NEG_INFINITY, 0);
    for (candidate, value) in values.iter().copied().enumerate() {
        if value > top {
            second = top;
            top = value;
            index = candidate;
        } else if value > second {
            second = value;
        }
    }
    (u32::try_from(index).unwrap_or_default(), top - second, top)
}
