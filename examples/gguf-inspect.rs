//! Prints a GGUF model's metadata, tensor types and directory without reading
//! tensor payloads. `--prefix` accepts a partial file, such as a download in
//! progress or a fetched header range.
#![allow(clippy::print_stdout)]

use std::{env, fs, path::PathBuf};

use libmir::gguf::{GgufHeader, GgufModel};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn main() -> Result<()> {
    let mut arguments = env::args().skip(1);
    let first = arguments.next().ok_or("usage: gguf-inspect [--prefix] FILE.gguf")?;
    let (prefix, path) = if first == "--prefix" {
        (true, PathBuf::from(arguments.next().ok_or("missing FILE.gguf")?))
    } else {
        (false, PathBuf::from(first))
    };
    let (header, tensors) = if prefix {
        let bytes = fs::read(&path)?;
        let header = GgufHeader::parse(&bytes, None).map_err(|error| format!("{error:?}"))?;
        let tensors = header.tensors.clone();
        (header, tensors)
    } else {
        let model = GgufModel::open(&path)?;
        println!("files: {}", model.shards.len());
        let tensors = model.tensors().cloned().collect();
        (model.header().clone(), tensors)
    };
    println!(
        "GGUF v{} · {} tensors · data at byte {}",
        header.version,
        tensors.len(),
        header.data_start
    );
    for (key, value) in &header.metadata {
        println!("  {key} = {}", value.summary());
    }
    let mut totals = std::collections::BTreeMap::new();
    for tensor in &tensors {
        let entry = totals.entry(tensor.ggml_type).or_insert((0_u64, 0_usize));
        entry.0 += tensor.bytes;
        entry.1 += 1;
    }
    println!("types:");
    for (kind, (bytes, count)) in totals {
        println!(
            "  {kind:8} {count:>4} tensors {:>9.3} GiB  {:>6.3} bits/weight",
            gib(bytes),
            kind.bits_per_weight()
        );
    }
    println!("tensors (first 16 by directory order):");
    for tensor in tensors.iter().take(16) {
        println!("  {:40} {:8} {:?}", tensor.name, tensor.ggml_type.name(), tensor.shape()?);
    }
    Ok(())
}

fn gib(bytes: u64) -> f64 {
    let mib = u32::try_from(bytes >> 20).unwrap_or(u32::MAX);
    f64::from(mib) / 1024.0
}
