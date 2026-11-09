use std::{fs, path::Path, sync::Arc};

use models::{
    execution::DecoderExecutionContract,
    layout::{DecoderConfig, ModelLayout},
    weights::TensorCatalog,
};

use super::super::{
    DenseSwiGluModel,
    tests::{write_config, write_weights},
};
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
        std::env::temp_dir().join(format!("libmir-metal-dense-history-{}", std::process::id()));
    fs::create_dir_all(&root)?;
    write_config(&root)?;
    write_weights(&root.join("model.safetensors"), "BF16")?;
    let persistent = decode(&root, DenseHistoryMode::Persistent)?;
    let per_row = decode(&root, DenseHistoryMode::PerRow)?;
    fs::remove_dir_all(&root)?;

    assert_eq!(persistent.len(), per_row.len());
    for (index, (left, right)) in persistent.iter().zip(&per_row).enumerate() {
        assert!((left - right).abs() < 1.0e-2, "logit {index}: {left} != {right}");
    }
    Ok(())
}

fn decode(root: &Path, dense_history: DenseHistoryMode) -> Result<Vec<f32>> {
    let mut config = MetalConfig::default();
    config.batch.dense_history = dense_history;
    let stream = Stream::new_gpu_with_config(Arc::new(config))?;
    let layout = ModelLayout::inspect(root)?;
    let decoder = DecoderConfig::from_layout(&layout)?;
    let catalog = TensorCatalog::from_layout(&layout)?;
    let contract = DecoderExecutionContract::discover(&layout, &decoder, &catalog)?;
    let lowering = lowering::plan(&contract.semantic)?;
    // MLX reads SafeTensors payloads only on a CPU stream.
    let tensors = ModelTensors::load(root, &Stream::new_cpu()?)?;
    let model = DenseSwiGluModel::load(
        &tensors,
        &decoder,
        &contract.bindings,
        lowering.layers(),
        16,
        &stream,
    )?;
    let mut first = model.new_cache(&stream)?;
    let mut second = model.new_cache(&stream)?;
    let mut logits = Vec::new();
    for step in 0..STEPS {
        let tokens = Array::from_u32(&[step % 4, (step + 1) % 4], &[2, 1])?;
        let position = i32::try_from(step)?;
        let output = model.forward_packed_decode(
            &tokens,
            &mut [&mut first, &mut second],
            &[position, position],
            &stream,
        )?;
        stream.eval_many_with_paged_arenas(&[&output])?;
        logits.extend(output.to_vec_f32(&stream)?);
    }
    Ok(logits)
}
