//! Split models map their names by shard 1's metadata across every shard.

use super::{GgmlType, GgufModel, Result, Writer};

fn shard(number: u32, writer: Writer) -> Writer {
    let unsigned = |value: u32| value.to_le_bytes().to_vec();
    writer.kv("split.no", 4, unsigned(number)).kv("split.count", 4, unsigned(2))
}

/// Shard 2 of a Nemotron-H model has no `general.architecture`, yet its
/// llama.cpp `ssm_a` still maps to the checkpoint's `A_log` with its shape.
#[test]
fn later_shards_map_by_the_first_shards_metadata() -> Result<()> {
    let directory = std::env::temp_dir().join(format!("libmir-gguf-split-{}", std::process::id()));
    std::fs::create_dir_all(&directory)?;
    let first = shard(0, Writer::new().text("general.architecture", "nemotron_h")?)
        .kv("split.tensors.count", 4, 2_u32.to_le_bytes().to_vec())
        .tensor("blk.0.ssm_d", &[1, 4], GgmlType::F32)?;
    let second = shard(1, Writer::new()).tensor("blk.1.ssm_a", &[1, 4], GgmlType::F32)?;
    let paths = [1, 2].map(|number| directory.join(format!("m-{number:05}-of-00002.gguf")));
    std::fs::write(&paths[0], first.bytes()?)?;
    std::fs::write(&paths[1], second.bytes()?)?;
    let catalog = GgufModel::open(&paths[1])?.mapped_catalog()?;
    std::fs::remove_dir_all(&directory)?;
    let decay = catalog.get("backbone.layers.1.mixer.A_log");
    assert_eq!(decay.map(|tensor| tensor.shape.clone()), Some(vec![4]));
    assert!(catalog.get("backbone.layers.0.mixer.D").is_some());
    Ok(())
}

/// Headers fetched elsewhere, such as by range requests, build the same model
/// as the files they came from; out-of-order shards are refused.
#[test]
fn headers_from_elsewhere_build_the_same_model() -> Result<()> {
    let first = shard(0, Writer::new().text("general.architecture", "nemotron_h")?)
        .kv("split.tensors.count", 4, 2_u32.to_le_bytes().to_vec())
        .tensor("blk.0.ssm_d", &[1, 4], GgmlType::F32)?
        .bytes()?;
    let second = shard(1, Writer::new()).tensor("blk.1.ssm_a", &[1, 4], GgmlType::F32)?.bytes()?;
    let header = |bytes: &[u8]| {
        crate::gguf::GgufHeader::parse(bytes, None)
            .map_err(|_| crate::error::ModelsError::InvalidGguf("unparsed test header".into()))
    };
    let paths =
        [1, 2].map(|number| std::path::PathBuf::from(format!("m-{number:05}-of-00002.gguf")));
    let shards = vec![(paths[0].clone(), header(&first)?), (paths[1].clone(), header(&second)?)];
    let catalog = GgufModel::from_headers(shards.clone())?.mapped_catalog()?;
    assert!(catalog.get("backbone.layers.1.mixer.A_log").is_some());
    let reversed = shards.into_iter().rev().collect::<Vec<_>>();
    assert!(GgufModel::from_headers(reversed).is_err());
    assert!(GgufModel::from_headers(Vec::new()).is_err());
    Ok(())
}
