//! Imports a GGUF model into a directory that loads like a `SafeTensors`
//! checkpoint: derived `config.json`, `generation_config.json`,
//! `tokenizer.json` and `tokenizer_config.json`, plus links to the GGUF files.
//! An `mmproj` file adds the vision tower and `preprocessor_config.json`.
#![allow(clippy::print_stdout)]

use std::{env, path::PathBuf};

use libmir::gguf::{GgufModel, Import};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn main() -> Result<()> {
    let mut arguments = env::args().skip(1);
    let usage = "usage: gguf-import FILE.gguf DIRECTORY [MMPROJ.gguf]";
    let source = PathBuf::from(arguments.next().ok_or(usage)?);
    let directory = PathBuf::from(arguments.next().ok_or(usage)?);
    let model = GgufModel::open(&source)?;
    let mut import = Import::derive(&model)?;
    if let Some(mmproj) = arguments.next() {
        import = import.with_vision(&GgufModel::open(mmproj)?, &model)?;
    }
    import.write(&directory)?;
    println!(
        "imported {} ({} files) into {}",
        source.display(),
        import.shards.len(),
        directory.display()
    );
    Ok(())
}
