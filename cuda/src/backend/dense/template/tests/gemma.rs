use std::{fs, path::Path};

use mircuda::bf16;
use models::{
    layout::{DecoderConfig, ModelLayout},
    semantic::SemanticModelSpec,
    weights::{TensorCatalog, WeightBindingPlan},
};
use runtime::kv::{BlockId, BlockTable, CacheConfig, KvCacheDType};

use super::{TestResult, copy, read};
use crate::{CudaBackend, CudaConfig, DenseSwiGluLayerLoadConfig, ProjectionFormat};

/// Teacher-forced per-layer comparison with
/// `benchmarks/tools/gemma/layer_oracle.py`: every layer reads the reference
/// input, so each error belongs to one layer.
#[test]
#[ignore = "requires GEMMA_DENSE_MODEL and GEMMA_LAYER_ORACLE"]
fn dense_gemma_layers_match_reference() -> TestResult<()> {
    let root = std::env::var("GEMMA_DENSE_MODEL")?;
    let oracle = std::env::var("GEMMA_LAYER_ORACLE")?;
    let oracle = Path::new(&oracle);
    let meta: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(oracle.join("meta.json"))?)?;
    let tokens = usize::try_from(meta["tokens"].as_u64().ok_or("meta lacks tokens")?)?;
    let layers = usize::try_from(meta["layers"].as_u64().ok_or("meta lacks layers")?)?;
    let layout = ModelLayout::inspect(Path::new(&root))?;
    let decoder = DecoderConfig::from_layout(&layout)?;
    let catalog = TensorCatalog::from_layout(&layout)?;
    let spec = SemanticModelSpec::from_layout(&layout, &decoder, &catalog)?;
    let bindings = WeightBindingPlan::discover_from_layout(&spec, &catalog, &layout)?;
    let backend = CudaBackend::new(CudaConfig::default())?;
    let load = DenseSwiGluLayerLoadConfig {
        cache: CacheConfig {
            block_size: 16,
            block_count: 1,
            dtype: KvCacheDType::BFloat16,
        },
        max_sequence_blocks: 1,
        projection_format: ProjectionFormat::NvFp4,
    };
    let hidden = decoder.hidden_size;
    let mut worst: f64 = 0.0;
    let mut errors = Vec::new();
    for layer in 0..layers {
        let input_name = if layer == 0 {
            "input.bf16".into()
        } else {
            format!("layer{}.bf16", layer - 1)
        };
        let input = copy(&backend, &values(&oracle.join(input_name))?)?;
        let expected = values(&oracle.join(format!("layer{layer}.bf16")))?;
        let (template, _) = backend.load_dense_swiglu_layer_tracked(
            &decoder,
            &spec,
            &catalog,
            layer,
            bindings.dense_decoder_layer(layer)?,
            load,
        )?;
        let config = template.config();
        let single = backend.inner.pool.allocate::<bf16>(&backend.inner.stream, hidden)?;
        let cache = backend.prepare_paged_kv(layer, config.attention.cache)?;
        let mut state = template.instantiate_with_cache(&single, &single, cache)?;
        let mut prefill = template.instantiate_prefill(tokens)?;
        let mut table = BlockTable::with_block_size(16);
        table.push(BlockId(0));
        table.set_token_len(tokens);
        let mut batch =
            backend.prepare_paged_prefill_batch(config.attention.cache, 1, 1, tokens)?;
        batch.prepare(&[&table], &[0], &[tokens])?;
        let mut output =
            backend.inner.pool.allocate::<bf16>(&backend.inner.stream, tokens * hidden)?;
        prefill.execute_batch(&mut state, &input, template.weights(), &batch, &mut output)?;
        let error = relative_l2(&read(&backend, &output)?, &expected);
        errors.push(format!("{layer} {:?}: {error:.4}", decoder.layer_type(layer)));
        worst = worst.max(error);
    }
    assert!(worst < 0.1, "worst layer relative L2 {worst:.4}; per layer {errors:?}");
    Ok(())
}

fn values(path: &Path) -> TestResult<Vec<bf16>> {
    Ok(fs::read(path)?
        .as_chunks::<2>()
        .0
        .iter()
        .map(|bytes| bf16::from_bits(u16::from_le_bytes(*bytes)))
        .collect())
}

fn relative_l2(actual: &[bf16], expected: &[bf16]) -> f64 {
    let (difference, norm) = actual.iter().zip(expected).fold((0.0, 0.0), |(d, n), (a, e)| {
        let (a, e) = (f64::from(a.to_f32()), f64::from(e.to_f32()));
        ((a - e).mul_add(a - e, d), e.mul_add(e, n))
    });
    (difference / norm).sqrt()
}
