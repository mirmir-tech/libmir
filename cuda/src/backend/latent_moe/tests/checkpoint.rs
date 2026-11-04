//! V3 gate: one real latent routed-expert block against an FP32 reference
//! with exactly dequantized NVFP4 experts, written by
//! `benchmarks/tools/nemotron/moe_oracle.py`.

use std::{
    fs,
    path::{Path, PathBuf},
};

use mircuda::bf16;
use models::{
    execution::DecoderExecutionContract,
    layout::{DecoderConfig, ModelLayout},
    weights::TensorCatalog,
};

use super::{copy, read};
use crate::{
    CudaConfig, CudaLatentMoe, Error, LatentMoeConfig, LatentMoeWeights, Result,
    backend::CudaBackend,
};

fn bf16s(directory: &Path, name: &str) -> Result<Vec<bf16>> {
    let bytes = fs::read(directory.join(name))?;
    Ok(bytes.as_chunks::<2>().0.iter().map(|pair| bf16::from_le_bytes(*pair)).collect())
}

#[test]
#[ignore = "requires NEMOTRON_MODEL and NEMOTRON_MOE_ORACLE"]
fn matches_the_reference_on_a_real_latent_block() -> Result<()> {
    let (Some(model), Some(oracle)) = (
        std::env::var_os("NEMOTRON_MODEL").map(PathBuf::from),
        std::env::var_os("NEMOTRON_MOE_ORACLE").map(PathBuf::from),
    ) else {
        return Ok(());
    };
    let meta: serde_json::Value = serde_json::from_slice(&fs::read(oracle.join("meta.json"))?)
        .map_err(|_| Error::InvalidExecutionPlan("oracle meta.json is not valid JSON"))?;
    let logical = meta["logical_layer"]
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
        .ok_or(Error::InvalidExecutionPlan("oracle meta has no logical layer"))?;
    let layout = ModelLayout::inspect(&model)?;
    let decoder = DecoderConfig::from_layout(&layout)?;
    let catalog = TensorCatalog::from_layout(&layout)?;
    let contract = DecoderExecutionContract::discover(&layout, &decoder, &catalog)?;
    let routed = contract
        .bindings
        .state_space_hybrid_layer(logical)?
        .feed_forward
        .ok_or(Error::InvalidExecutionPlan("oracle layer has no routed block"))?;
    let hybrid = decoder
        .state_space
        .as_ref()
        .ok_or(Error::InvalidExecutionPlan("model is not a state-space hybrid"))?;
    let config = LatentMoeConfig::from_hybrid(decoder.hidden_size, &hybrid.routed);
    let backend = CudaBackend::new(CudaConfig::default())?;
    let mut upload = backend.begin_tensor_upload();
    let mut dense =
        vec![routed.router, routed.correction_bias, routed.latent_input, routed.latent_output];
    if let Some(shared) = routed.shared {
        dense.extend([shared.up, shared.down]);
    }
    for name in dense.iter().flat_map(|binding| binding.physical_sources()) {
        upload.enqueue(catalog.get(name).ok_or_else(|| Error::MissingTensor(name.into()))?)?;
    }
    let tensors = upload.finish()?;
    let weights = LatentMoeWeights::load_bindings(&backend, &tensors, &catalog, routed, config)?;
    let block = CudaLatentMoe::new(&backend, config, weights)?;

    let input = bf16s(&oracle, "input.bf16")?;
    let expected = bf16s(&oracle, "output.bf16")?;
    let tokens = input.len() / config.hidden;
    let mut execution = block.prepare(tokens)?;
    let mut output = backend.pool().allocate(backend.stream(), expected.len())?;
    let input = copy(&backend, &input)?;
    execution.execute(&input, &mut output)?;
    let actual = read(&backend, &output)?;
    assert_parity(&actual, &expected, config.hidden);
    // Repeats must be bit-identical: the block has no timing-dependent or
    // order-dependent reduction.
    for _ in 0..3 {
        execution.execute(&input, &mut output)?;
        let repeat = read(&backend, &output)?;
        let differing = actual.iter().zip(&repeat).filter(|(a, b)| a.to_bits() != b.to_bits());
        assert_eq!(differing.count(), 0, "repeated latent MoE block differs");
    }
    Ok(())
}

/// Relative L2 error below 3% and cosine similarity of every token of at
/// least 0.999. Marlin computes the routed experts as W4A16 with BF16
/// intermediates; the shared expert's W4A4 activation rounding differs from
/// the reference emulation in block-scale details.
fn assert_parity(actual: &[bf16], expected: &[bf16], row: usize) {
    assert_eq!(actual.len(), expected.len());
    let (mut error, mut norm) = (0.0, 0.0);
    let mut worst_cosine = 1.0_f64;
    for (actual, expected) in actual.chunks(row).zip(expected.chunks(row)) {
        let (mut dot, mut left, mut right) = (0.0, 0.0, 0.0);
        for (actual, expected) in actual.iter().zip(expected) {
            let (actual, expected) = (f64::from(actual.to_f32()), f64::from(expected.to_f32()));
            error += (actual - expected).powi(2);
            norm += expected.powi(2);
            dot = actual.mul_add(expected, dot);
            left = actual.mul_add(actual, left);
            right = expected.mul_add(expected, right);
        }
        if left > 0.0 && right > 0.0 {
            worst_cosine = worst_cosine.min(dot / (left.sqrt() * right.sqrt()));
        }
    }
    let l2 = (error / norm.max(f64::MIN_POSITIVE)).sqrt();
    assert!(
        l2 < 0.03 && worst_cosine >= 0.999,
        "relative L2 {l2}, worst token cosine {worst_cosine}"
    );
}
