use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

use models::{
    execution::DecoderExecutionContract,
    layout::{DecoderConfig, ModelLayout},
    weights::TensorCatalog,
};

use super::StateSpaceHybridModel;
use crate::engine::{Array, DecoderCache, ModelTensors, Result, Stream, lowering};

mod fixture;
mod multimodal;
mod oracle;
mod restore;

use fixture::{HIDDEN, write_checkpoint};

static SEQUENCE: AtomicUsize = AtomicUsize::new(0);

struct Harness {
    model: StateSpaceHybridModel,
    stream: Stream,
    _tensors: ModelTensors,
    root: PathBuf,
}

impl Drop for Harness {
    fn drop(&mut self) {
        drop(fs::remove_dir_all(&self.root));
    }
}

fn harness(latent: Option<usize>) -> Result<Harness> {
    let root = std::env::temp_dir().join(format!(
        "libmir-metal-state-space-{}-{}",
        std::process::id(),
        SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&root)?;
    write_checkpoint(&root, latent)?;
    let layout = ModelLayout::inspect(&root)?;
    let decoder = DecoderConfig::from_layout(&layout)?;
    let catalog = TensorCatalog::from_layout(&layout)?;
    let contract = DecoderExecutionContract::discover(&layout, &decoder, &catalog)?;
    let lowering = lowering::plan(&decoder, &contract.semantic)?;
    let load_stream = Stream::new_cpu()?;
    let tensors = ModelTensors::load(&root, &load_stream)?;
    let stream = Stream::new_gpu()?;
    let model = StateSpaceHybridModel::load(
        &tensors,
        &decoder,
        &contract.semantic.decoder,
        &contract.bindings,
        lowering.layers(),
        16,
        &stream,
    )?;
    Ok(Harness { model, stream, _tensors: tensors, root })
}

fn tokens(row: u32, count: usize) -> Vec<u32> {
    (0..count)
        .map(|index| row.wrapping_mul(7).wrapping_add(u32::try_from(index).unwrap_or(0)) % 64)
        .collect()
}

fn assert_close(actual: &[f32], expected: &[f32], relative: f32) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        let tolerance = relative * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual {actual}, expected {expected}, tolerance {tolerance}"
        );
    }
}

fn step_tokens(
    harness: &Harness,
    cache: &mut DecoderCache,
    tokens: &[u32],
    start: i32,
) -> Result<Vec<f32>> {
    let mut logits = Vec::new();
    for (position, token) in tokens.iter().enumerate() {
        let output = harness.model.forward_decode(
            &Array::from_u32(&[*token], &[1, 1])?,
            cache,
            start + i32::try_from(position)?,
            &harness.stream,
        )?;
        logits = output.to_vec_f32(&harness.stream)?;
    }
    Ok(logits)
}

fn executes(latent: Option<usize>) -> Result<()> {
    let harness = harness(latent)?;
    let mut cache = harness.model.new_cache(&harness.stream)?;
    let logits = harness.model.forward_decode(
        &Array::from_u32(&[3], &[1, 1])?,
        &mut cache,
        0,
        &harness.stream,
    )?;
    assert_eq!(logits.shape()?, vec![1, 1, 64]);
    assert!(logits.to_vec_f32(&harness.stream)?.iter().all(|value| value.is_finite()));
    Ok(())
}

#[test]
fn executes_stacked_experts_without_a_latent_space() -> Result<()> {
    executes(None)
}

#[test]
fn executes_stacked_experts_in_a_latent_space() -> Result<()> {
    executes(Some(16))
}

#[test]
fn chunked_prefill_matches_token_by_token_recurrence() -> Result<()> {
    let harness = harness(None)?;
    let prompt = tokens(1, 21);
    let mut cache = harness.model.new_cache(&harness.stream)?;
    let last = step_tokens(&harness, &mut cache, &prompt, 0)?;
    assert_eq!(cache.cached_tokens()?, prompt.len());

    let mut prefilled = harness.model.new_cache(&harness.stream)?;
    let (head, tail) = prompt.split_at(prompt.len() - 1);
    let hidden = harness.model.forward_prefill(
        &Array::from_u32(head, &[1, i32::try_from(head.len())?])?,
        &mut prefilled,
        0,
        &harness.stream,
    )?;
    hidden.async_eval(&harness.stream)?;
    assert_eq!(prefilled.cached_tokens()?, head.len());
    let logits = step_tokens(&harness, &mut prefilled, tail, i32::try_from(head.len())?)?;
    // The chunked closed form and the per-token recurrence accumulate floating
    // point error differently; a tiny miniature keeps them within 1e-2.
    assert_close(&logits, &last, 1e-2);
    Ok(())
}

#[test]
fn packed_prefill_matches_independent_rows() -> Result<()> {
    let harness = harness(Some(16))?;
    let rows = [tokens(2, 9), tokens(3, 9)];
    let mut caches =
        [harness.model.new_cache(&harness.stream)?, harness.model.new_cache(&harness.stream)?];
    let expected = rows
        .iter()
        .zip(&mut caches)
        .map(|(row, cache)| {
            harness.model.forward_prefill(
                &Array::from_u32(row, &[1, 9])?,
                cache,
                0,
                &harness.stream,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let expected = Array::concatenate(&expected.iter().collect::<Vec<_>>(), 0, &harness.stream)?;
    let mut packed =
        [harness.model.new_cache(&harness.stream)?, harness.model.new_cache(&harness.stream)?];
    let mut refs = packed.iter_mut().collect::<Vec<_>>();
    let actual = harness.model.forward_packed_prefill_state(
        &Array::from_u32(&rows.concat(), &[2, 9])?,
        &mut refs,
        &[0, 0],
        &harness.stream,
    )?;
    assert_close(
        &actual.to_vec_f32(&harness.stream)?,
        &expected.to_vec_f32(&harness.stream)?,
        1e-3,
    );

    let token = [5, 6];
    let scalar = rows
        .iter()
        .zip(&mut caches)
        .zip(token)
        .map(|((_, cache), token)| {
            harness.model.forward_decode(
                &Array::from_u32(&[token], &[1, 1])?,
                cache,
                9,
                &harness.stream,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let scalar = Array::concatenate(&scalar.iter().collect::<Vec<_>>(), 0, &harness.stream)?;
    let mut refs = packed.iter_mut().collect::<Vec<_>>();
    let decoded = harness.model.forward_packed_decode(
        &Array::from_u32(&token, &[2, 1])?,
        &mut refs,
        &[9, 9],
        &harness.stream,
    )?;
    assert_close(
        &decoded.to_vec_f32(&harness.stream)?,
        &scalar.to_vec_f32(&harness.stream)?,
        1e-3,
    );
    Ok(())
}

#[test]
fn snapshot_restores_recurrent_state() -> Result<()> {
    let harness = harness(None)?;
    let prompt = tokens(4, 10);
    let mut cache = harness.model.new_cache(&harness.stream)?;
    let hidden = harness.model.forward_prefill(
        &Array::from_u32(&prompt, &[1, 10])?,
        &mut cache,
        0,
        &harness.stream,
    )?;
    let mut roots = vec![&hidden];
    cache.extend_prefill_graph_roots(&mut roots);
    harness.stream.eval_many_with_paged_arenas(&roots)?;
    let mut snapshot = cache.snapshot_at(10)?;
    let continuation = tokens(5, 3);
    let original = step_tokens(&harness, &mut cache, &continuation, 10)?;
    let restored = step_tokens(&harness, &mut snapshot, &continuation, 10)?;
    assert_close(&restored, &original, 1e-5);
    cache.reset()?;
    assert_eq!(cache.cached_tokens()?, 0);
    assert_eq!(hidden.shape()?, vec![1, 10, i32::try_from(HIDDEN)?]);
    Ok(())
}
