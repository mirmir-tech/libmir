//! Builds a generation contract from `config.json` and a directory of
//! `SafeTensors` header prefixes, then reports CUDA and Metal admission.
//!
//! Each `*.head` file holds the first `8 + header_len` bytes of one shard, so
//! large checkpoints can be validated before their payloads are downloaded.
#![allow(clippy::print_stdout)]

use std::{env, fs, path::PathBuf};

use libmir::{BackendTarget, Error, RemoteModelContract, TensorCatalog, TensorInfo};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn main() -> Result<()> {
    let config_path = argument(1, "config.json path")?;
    let headers = argument(2, "header directory")?;
    let config: serde_json::Value = serde_json::from_str(&fs::read_to_string(&config_path)?)?;
    let catalog = catalog(&headers)?;
    println!("tensors: {}", catalog.len());
    let contract = RemoteModelContract::inspect_generation(&config, &catalog)?;
    if let Some(execution) = contract.execution() {
        println!("semantic layers: {}", execution.semantic.decoder.layers.len());
        println!("prediction modules: {}", execution.semantic.decoder.prediction_layers.len());
        println!("bindings: {}", execution.bindings.tensors.len());
    }
    println!("encoding: {:?}", contract.checkpoint_encoding());
    println!("requirements: {:?}", contract.architecture_requirements().capabilities);
    for backend in [BackendTarget::Cuda, BackendTarget::Metal] {
        let report = contract.admission(backend.clone());
        println!("{backend:?}: {:?}", report.status);
        for check in &report.checks {
            println!("  {check:?}");
        }
    }
    Ok(())
}

fn catalog(directory: &PathBuf) -> Result<TensorCatalog> {
    let mut tensors: Vec<TensorInfo> = Vec::new();
    let mut entries = fs::read_dir(directory)?.collect::<std::io::Result<Vec<_>>>()?;
    entries.sort_by_key(fs::DirEntry::file_name);
    for entry in entries {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let Some(shard) = name.strip_suffix(".head") else {
            continue;
        };
        let bytes = fs::read(&path)?;
        tensors.extend(TensorCatalog::from_safetensors_header(shard, &bytes)?.tensors);
    }
    Ok(TensorCatalog::new(tensors))
}

fn argument(index: usize, name: &'static str) -> Result<PathBuf> {
    Ok(env::args_os()
        .nth(index)
        .map(PathBuf::from)
        .ok_or(Error::MissingEnvironment(name))?)
}
