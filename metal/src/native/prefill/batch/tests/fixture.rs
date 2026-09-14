use std::{path::PathBuf, sync::Arc};

use foundation::model::{BackendTarget, ModelManifest, Quantization};
use runtime::tuning::TuningMode;
use uuid::Uuid;

use super::{LoadedModel, Result};

pub(super) fn load() -> Result<(LoadedModel, PathBuf)> {
    load_family(Family::Clamped, 0)
}

#[derive(Clone, Copy, Debug)]
pub(super) enum Family {
    Clamped,
    Hybrid,
}

pub(super) fn load_family(family: Family, prefixes: usize) -> Result<(LoadedModel, PathBuf)> {
    let directory = std::env::temp_dir().join(format!("libmir-prefill-budget-{}", Uuid::new_v4()));
    std::fs::create_dir_all(&directory)?;
    match family {
        Family::Clamped => {
            use crate::engine::clamped_routed::tests::fixture::{write_config, write_weights};
            write_config(&directory)?;
            write_weights(&directory.join("model.safetensors"), true)?;
        },
        Family::Hybrid => {
            use crate::engine::hybrid_linear_moe::tests::{write_config, write_weights};
            write_config(&directory)?;
            write_weights(&directory.join("model.safetensors"))?;
        },
    }
    let model = load_path(directory.to_string_lossy().into_owned(), prefixes)?;
    Ok((model, directory))
}

pub(super) fn load_path(path: String, prefixes: usize) -> Result<LoadedModel> {
    load_path_with_context(path, prefixes, 256)
}

pub(super) fn load_path_with_context(
    path: String,
    prefixes: usize,
    context_len: usize,
) -> Result<LoadedModel> {
    load_path_with_paging(path, prefixes, context_len, 1)
}

pub(super) fn load_path_with_paging(
    path: String,
    prefixes: usize,
    context_len: usize,
    page_min_context: usize,
) -> Result<LoadedModel> {
    let manifest = ModelManifest {
        id: "prefill-budget-test".into(),
        path,
        tokenizer_path: None,
        context_len,
        preferred_backends: vec![BackendTarget::Metal],
        quantization: Quantization::Int4,
    };
    let mut config = crate::MetalConfig::default();
    config.cache.prefix_cache_entries = prefixes;
    config.cache.paged_attention_min_context = page_min_context;
    config.tuning.mode = TuningMode::Disabled;
    LoadedModel::load_with_config(&manifest, Arc::new(config), &mut |_| {})
}
