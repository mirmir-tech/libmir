use std::{path::PathBuf, sync::Arc};

use foundation::model::{BackendTarget, ModelManifest, Quantization};
use runtime::{backend::SamplingLogits, tuning::TuningMode};
use uuid::Uuid;

use super::*;
use crate::engine::{Array, DecoderCache, KvCache, KvPageFormat};

type TestResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[test]
fn rejects_prefill_when_a_later_layer_has_no_free_pages() -> TestResult<()> {
    let (mut model, directory) = load()?;
    let maximum = DecoderCache::physical_page_capacity(model.stream(), model.info.cache_step);
    let mut blocker = occupy(&model, 1, maximum)?;
    assert!(model.reserve_prefill_pages(1).is_err(), "only the first layer was checked");
    let session = Uuid::new_v4();
    let mut progress = 0;
    assert!(
        model
            .prefill(session, &[1, 2], &[], SamplingLogits::Full, None, &mut |_| {
                progress += 1;
            })
            .is_err()
    );
    assert_eq!(progress, 0, "prefill started before admission failed");
    assert!(!model.sessions.contains_key(&session));
    assert_eq!(available(&model, 0, maximum)?, maximum);
    blocker.reset()?;
    model.reserve_prefill_pages(1)?;
    drop(model.prefill(session, &[1, 2], &[], SamplingLogits::Full, None, &mut |_| {})?);
    model.release_session(session)?;
    model.clear_prefix_cache();
    assert_eq!(model.stream().paged_arenas().resident_arenas()?, 0);
    drop(model);
    std::fs::remove_dir_all(directory)?;
    Ok(())
}

fn available(model: &LoadedModel, layer: usize, maximum: usize) -> TestResult<usize> {
    Ok(model
        .stream()
        .paged_arenas()
        .available_pages(maximum, layer, 2, 8, KvPageFormat::Native)?)
}

#[test]
fn evicts_prefix_for_a_later_layer_and_rechecks_headroom() -> TestResult<()> {
    let (mut model, directory) = load()?;
    let session = Uuid::new_v4();
    drop(model.prefill(session, &[1, 2], &[], SamplingLogits::Full, None, &mut |_| {})?);
    model.release_session(session)?;
    assert_eq!(model.prefixes.group_count(), 1);
    let maximum = DecoderCache::physical_page_capacity(model.stream(), model.info.cache_step);
    assert_eq!(available(&model, 0, maximum)?, maximum - 1);
    let mut blocker = occupy(&model, 1, maximum - 1)?;
    assert_eq!(available(&model, 1, maximum)?, 0);
    model.reserve_prefill_pages(0)?;
    assert_eq!(model.prefixes.group_count(), 1, "no request must not evict a prefix");
    model.reserve_prefill_pages(1)?;
    assert_eq!(model.prefixes.group_count(), 0, "later-layer pressure did not evict the prefix");
    assert_eq!(available(&model, 1, maximum)?, 1);
    blocker.reset()?;
    assert_eq!(model.stream().paged_arenas().resident_arenas()?, 0);
    drop(model);
    std::fs::remove_dir_all(directory)?;
    Ok(())
}

fn occupy(model: &LoadedModel, layer: usize, pages: usize) -> TestResult<KvCache> {
    let size = model.stream().config().kv_cache.block_size;
    let mut cache = KvCache::new_paged_with_pool_capacity(
        size,
        size,
        KvPageFormat::Native,
        DecoderCache::physical_page_capacity(model.stream(), model.info.cache_step),
        Arc::clone(model.stream().paged_arenas()),
        layer,
    )?;
    let values = Array::from_f32(
        &vec![0.0; 2 * pages * size * 8],
        &[1, 2, i32::try_from(pages * size)?, 8],
    )?;
    drop(cache.update(&values, &values, model.stream())?);
    model.stream().paged_arenas().eval_with_graph_roots(&[], model.stream())?;
    Ok(cache)
}

fn load() -> TestResult<(LoadedModel, PathBuf)> {
    use crate::engine::clamped_routed::tests::fixture::{write_config, write_weights};
    let directory = std::env::temp_dir().join(format!("libmir-layer-capacity-{}", Uuid::new_v4()));
    std::fs::create_dir_all(&directory)?;
    write_config(&directory)?;
    let config_path = directory.join("config.json");
    let mut config: serde_json::Value = serde_json::from_slice(&std::fs::read(&config_path)?)?;
    config["num_hidden_layers"] = 2.into();
    config["layer_types"] = serde_json::json!(["full_attention", "full_attention"]);
    std::fs::write(config_path, serde_json::to_vec(&config)?)?;
    let weights = directory.join("model.safetensors");
    write_weights(&weights, true)?;
    duplicate_layer(&weights)?;
    let manifest = ModelManifest {
        id: "layer-capacity".into(),
        path: directory.to_string_lossy().into_owned(),
        tokenizer_path: None,
        context_len: 256,
        preferred_backends: vec![BackendTarget::Metal],
        quantization: Quantization::Int4,
    };
    let mut config = crate::MetalConfig::default();
    config.cache.prefix_cache_entries = 4;
    config.cache.paged_attention_min_context = 1;
    config.kv_cache.block_count = 1;
    config.set_max_batch_requests(1);
    config.tuning.mode = TuningMode::Disabled;
    Ok((
        LoadedModel::load_with_config(&manifest, Arc::new(config), &mut |_| {})?,
        directory,
    ))
}

fn duplicate_layer(path: &std::path::Path) -> TestResult<()> {
    let bytes = std::fs::read(path)?;
    let size = usize::try_from(u64::from_le_bytes(bytes[..8].try_into()?))?;
    let mut header: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(&bytes[8..8 + size])?;
    let mut payload = bytes[8 + size..].to_vec();
    for (name, mut entry) in header.clone() {
        if !name.starts_with("model.layers.0.") {
            continue;
        }
        let offsets: Vec<usize> = serde_json::from_value(entry["data_offsets"].clone())?;
        let start = payload.len();
        payload.extend_from_slice(&bytes[8 + size + offsets[0]..8 + size + offsets[1]]);
        entry["data_offsets"] = serde_json::json!([start, payload.len()]);
        header.insert(name.replacen("model.layers.0.", "model.layers.1.", 1), entry);
    }
    let mut header = serde_json::to_vec(&header)?;
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut output = u64::try_from(header.len())?.to_le_bytes().to_vec();
    output.extend(header);
    output.extend(payload);
    std::fs::write(path, output)?;
    Ok(())
}
