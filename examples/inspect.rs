#![allow(clippy::print_stdout)]

use std::{env, path::PathBuf};

use libmir::{Error, GenerationOverrides, ModelDescriptor};

fn main() -> libmir::Result<()> {
    let descriptor = ModelDescriptor::inspect(model_path()?, GenerationOverrides::default())?;
    let manifest = descriptor.manifest()?;

    println!("id: {}", manifest.id);
    println!("family: {:?}", manifest.family);
    println!("context: {}", manifest.context_len);
    println!("quantization: {:?}", manifest.quantization);
    println!("decoder: {:?}", descriptor.decoder());
    println!("generation: {:?}", descriptor.generation());
    Ok(())
}

fn model_path() -> libmir::Result<PathBuf> {
    env::args_os()
        .nth(1)
        .or_else(|| env::var_os("MODEL"))
        .map(PathBuf::from)
        .ok_or(Error::MissingEnvironment("MODEL or the first argument"))
}
