#![allow(clippy::print_stdout)]

use std::{env, path::PathBuf};

use libmir::{BackendTarget, Error, GenerationOverrides, ModelDescriptor, RuntimeConfig};

fn main() -> libmir::Result<()> {
    let descriptor = ModelDescriptor::inspect(model_path()?, GenerationOverrides::default())?;
    let manifest = descriptor.manifest()?;

    println!("id: {}", manifest.id);
    println!("context: {}", manifest.context_len);
    println!("quantization: {:?}", manifest.quantization);
    println!("metadata: {:?}", descriptor.metadata());
    println!("checkpoint encoding: {:?}", descriptor.checkpoint_encoding());
    println!("decoder: {:?}", descriptor.decoder());
    println!("generation: {:?}", descriptor.generation());
    println!("unloaded tensors: {:?}", descriptor.unloaded_tensors());
    println!(
        "memory estimate (CUDA, default runtime): {:?}",
        descriptor.memory_estimate_for(&RuntimeConfig::default(), &BackendTarget::Cuda)
    );
    Ok(())
}

fn model_path() -> libmir::Result<PathBuf> {
    env::args_os()
        .nth(1)
        .or_else(|| env::var_os("MODEL"))
        .map(PathBuf::from)
        .ok_or(Error::MissingEnvironment("MODEL or the first argument"))
}
