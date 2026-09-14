use std::fs;

use models::{
    execution::DecoderExecutionContract,
    layout::{DecoderConfig, ModelLayout},
    weights::TensorCatalog,
};
use serde_json::{Value, json};

pub mod fixture;
use fixture::{write_config, write_weights};

use super::ClampedRoutedModel;
use crate::{
    config::MoePrefill,
    engine::{Array, ModelTensors, Result, Stream, lowering},
};

#[test]
fn executes_a_complete_dense_clamped_routed_model() -> Result<()> {
    execute_dense_clamped_routed_model(false, false, false, MoePrefill::Default)
}

#[test]
fn executes_a_fused_dense_clamped_routed_model() -> Result<()> {
    execute_dense_clamped_routed_model(true, false, false, MoePrefill::Default)
}

#[test]
fn preserves_clamped_sliding_state_across_packed_execution() -> Result<()> {
    execute_dense_clamped_routed_model(true, true, false, MoePrefill::Default)
}

#[test]
fn profiling_preserves_clamped_packed_cache_and_logits() -> Result<()> {
    execute_dense_clamped_routed_model(true, true, true, MoePrefill::Default)
}

#[test]
fn sorted_clamped_prefill_preserves_packed_sliding_decode_continuation() -> Result<()> {
    for fused in [false, true] {
        execute_dense_clamped_routed_model(fused, true, false, MoePrefill::ClampedSorted)?;
    }
    Ok(())
}

fn execute_dense_clamped_routed_model(
    fused: bool,
    sliding: bool,
    profile: bool,
    moe: MoePrefill,
) -> Result<()> {
    let root = std::env::temp_dir().join(format!(
        "libmir-metal-dense-clamped-routed-{fused}-{sliding}-{profile}-{moe:?}-{}",
        std::process::id()
    ));
    fs::create_dir_all(&root)?;
    write_config(&root)?;
    if sliding {
        let path = root.join("config.json");
        let mut config: Value = serde_json::from_slice(&fs::read(&path)?)?;
        config["layer_types"] = json!(["sliding_attention"]);
        config["sliding_window"] = json!(4);
        fs::write(path, serde_json::to_vec(&config)?)?;
    }
    write_weights(&root.join("model.safetensors"), fused)?;

    let layout = ModelLayout::inspect(&root)?;
    let decoder = DecoderConfig::from_layout(&layout)?;
    let catalog = TensorCatalog::from_layout(&layout)?;
    let contract = DecoderExecutionContract::discover(&layout, &decoder, &catalog)?;
    let lowering = lowering::plan(&contract.semantic)?;
    let load_stream = Stream::new_cpu()?;
    let tensors = ModelTensors::load(&root, &load_stream)?;
    let mut config = crate::MetalConfig::default();
    config.diagnostics.profile_components = profile;
    config.diagnostics.moe_prefill = moe;
    let mut stream = Stream::new_gpu_with_config(std::sync::Arc::new(config))?;
    let model = ClampedRoutedModel::load(
        &tensors,
        &decoder,
        &contract.bindings,
        lowering.layers(),
        16,
        &stream,
    )?;
    let mut cache = model.new_cache(&stream)?;
    let logits = model.forward(&Array::from_u32(&[1], &[1, 1])?, &mut cache, 0, false, &stream)?;

    assert_eq!(logits.shape()?, vec![1, 1, 64]);
    assert!(logits.to_vec_f32(&stream)?.iter().all(|value| value.is_finite()));
    check_packed_state(&model, &stream)?;
    stream.set_rope_batching(crate::config::RopeBatching::Offsets);
    check_packed_state(&model, &stream)?;
    drop(tensors);
    fs::remove_dir_all(root)?;
    Ok(())
}

fn check_packed_state(model: &ClampedRoutedModel, stream: &Stream) -> Result<()> {
    let tokens = Array::from_u32(&[1, 2, 3, 4, 5, 6], &[1, 6])?;
    let mut reference = model.new_cache(stream)?;
    let full = model.forward(&tokens, &mut reference, 0, true, stream)?;
    full.async_eval(stream)?;
    stream.synchronize()?;
    let mut first = model.new_cache(stream)?;
    let mut second = model.new_cache(stream)?;
    let batch = Array::from_u32(&[1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1], &[2, 6])?;
    let hidden = model.forward_packed_state(
        &batch,
        &mut [&mut first, &mut second],
        &[0, 0],
        true,
        stream,
    )?;
    assert_eq!(hidden.shape()?, [2, 6, 32]);
    let mut roots = vec![&hidden];
    first.extend_graph_roots(&mut roots);
    second.extend_graph_roots(&mut roots);
    stream.eval_many_with_paged_arenas(&roots)?;
    let next = Array::from_u32(&[7], &[1, 1])?;
    let expected = model.forward(&next, &mut reference, 6, false, stream)?.to_vec_f32(stream)?;
    let capture = crate::engine::probe::routing::Capture::begin()?;
    let packed = model.forward_packed_decode(
        &Array::from_u32(&[7, 8], &[2, 1])?,
        &mut [&mut first, &mut second],
        &[6, 6],
        stream,
    )?;
    let routes = capture.finish()?;
    assert_eq!(routes.len(), 1);
    assert_eq!(routes[0].layer, 0);
    assert_eq!(routes[0].indices.shape()?, [2, 1]);
    let actual = packed.slice(&[0, 0, 0], &[1, 1, 64], stream)?.to_vec_f32(stream)?;
    assert!(
        actual
            .iter()
            .zip(&expected)
            .all(|(actual, expected)| (actual - expected).abs() < 1e-5)
    );
    let actual_second = packed.slice(&[1, 0, 0], &[2, 1, 64], stream)?.to_vec_f32(stream)?;
    let mut reference_second = model.new_cache(stream)?;
    let tokens = Array::from_u32(&[6, 5, 4, 3, 2, 1], &[1, 6])?;
    let full = model.forward(&tokens, &mut reference_second, 0, true, stream)?;
    let mut roots = vec![&full];
    reference_second.extend_graph_roots(&mut roots);
    stream.eval_many_with_paged_arenas(&roots)?;
    let next = Array::from_u32(&[8], &[1, 1])?;
    let expected_second = model
        .forward(&next, &mut reference_second, 6, false, stream)?
        .to_vec_f32(stream)?;
    assert!(
        actual_second
            .iter()
            .zip(&expected_second)
            .all(|(actual, expected)| (actual - expected).abs() < 1e-5)
    );
    Ok(())
}
