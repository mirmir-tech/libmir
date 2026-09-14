use std::{path::PathBuf, sync::Arc};

use foundation::model::{BackendTarget, ModelManifest, Quantization};
use runtime::tuning::TuningMode;

use super::{LoadedModel, Result};
use crate::engine::clamped_routed::tests::fixture::{write_config, write_weights};

pub(super) fn load() -> Result<(LoadedModel, PathBuf)> {
    let mut config = crate::MetalConfig::default();
    config.cache.prefix_cache_entries = 0;
    config.tuning.mode = TuningMode::Disabled;
    load_config(config)
}

pub(super) fn load_config(config: crate::MetalConfig) -> Result<(LoadedModel, PathBuf)> {
    let directory =
        std::env::temp_dir().join(format!("libmir-metal-mixed-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&directory)?;
    write_config(&directory)?;
    write_weights(&directory.join("model.safetensors"), true)?;
    let manifest = ModelManifest {
        id: "mixed-clamped-test".into(),
        path: directory.to_string_lossy().into_owned(),
        tokenizer_path: None,
        context_len: 256,
        preferred_backends: vec![BackendTarget::Metal],
        quantization: Quantization::Int4,
    };
    let model = LoadedModel::load_with_config(&manifest, Arc::new(config), &mut |_| {})?;
    Ok((model, directory))
}
