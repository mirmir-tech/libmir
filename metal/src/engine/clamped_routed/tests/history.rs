use std::{fs, path::Path, sync::Arc};

use models::{
    execution::DecoderExecutionContract,
    layout::{DecoderConfig, ModelLayout},
    weights::TensorCatalog,
};

use super::{ClampedRoutedModel, write_config, write_weights};
use crate::{
    MetalConfig,
    config::DenseHistoryMode,
    engine::{Array, ModelTensors, Result, Stream, lowering},
};

// Crosses the 128-token page promotion and one 256-token history growth.
const STEPS: u32 = 300;

#[test]
fn persistent_history_matches_per_row_decode() -> Result<()> {
    let root =
        std::env::temp_dir().join(format!("libmir-metal-clamped-history-{}", std::process::id()));
    fs::create_dir_all(&root)?;
    write_config(&root)?;
    write_weights(&root.join("model.safetensors"), false)?;
    // Zero keeps the rows aligned; a lead leaves them at different positions.
    for lead in [0, 1, 3, 7] {
        let persistent = decode(&root, DenseHistoryMode::Persistent, lead)?;
        let per_row = decode(&root, DenseHistoryMode::PerRow, lead)?;
        assert_eq!(persistent.len(), per_row.len());
        for (index, (left, right)) in persistent.iter().zip(&per_row).enumerate() {
            assert!((left - right).abs() < 1.0e-2, "lead {lead} logit {index}: {left} != {right}");
        }
    }
    fs::remove_dir_all(&root)?;
    Ok(())
}

fn decode(root: &Path, dense_history: DenseHistoryMode, lead: u32) -> Result<Vec<f32>> {
    let mut config = MetalConfig::default();
    config.batch.dense_history = dense_history;
    // Row-wise paged candidates are tuned per shape; keep the reference stable.
    config.tuning.mode = runtime::tuning::TuningMode::Disabled;
    let stream = Stream::new_gpu_with_config(Arc::new(config))?;
    let layout = ModelLayout::inspect(root)?;
    let decoder = DecoderConfig::from_layout(&layout)?;
    let catalog = TensorCatalog::from_layout(&layout)?;
    let contract = DecoderExecutionContract::discover(&layout, &decoder, &catalog)?;
    let lowering = lowering::plan(&contract.semantic)?;
    // MLX reads SafeTensors payloads only on a CPU stream.
    let tensors = ModelTensors::load(root, &Stream::new_cpu()?)?;
    let model = ClampedRoutedModel::load(
        &tensors,
        &decoder,
        &contract.bindings,
        lowering.layers(),
        16,
        &stream,
    )?;
    let mut first = model.new_cache(&stream)?;
    let mut second = model.new_cache(&stream)?;
    for step in 0..lead {
        let token = Array::from_u32(&[step % 4], &[1, 1])?;
        let output = model.forward(&token, &mut first, i32::try_from(step)?, false, &stream)?;
        stream.eval_many_with_paged_arenas(&[&output])?;
    }
    let mut logits = Vec::new();
    for step in 0..STEPS {
        let tokens = Array::from_u32(&[step % 4, (step + 1) % 4], &[2, 1])?;
        let position = i32::try_from(step)?;
        let leading = position + i32::try_from(lead)?;
        let output = model.forward_packed_decode(
            &tokens,
            &mut [&mut first, &mut second],
            &[leading, position],
            &stream,
        )?;
        let mut roots = vec![&output];
        first.extend_prefill_graph_roots(&mut roots);
        second.extend_prefill_graph_roots(&mut roots);
        stream.eval_many_with_paged_arenas(&roots)?;
        logits.extend(output.to_vec_f32(&stream)?);
    }
    Ok(logits)
}
