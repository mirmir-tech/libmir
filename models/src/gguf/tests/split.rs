//! Split models map their names by shard 1's metadata across every shard.

use super::{GgmlType, GgufModel, Result, ShardName, Writer};

fn shard(number: u32, writer: Writer) -> Writer {
    let unsigned = |value: u32| value.to_le_bytes().to_vec();
    writer.kv("split.no", 4, unsigned(number)).kv("split.count", 4, unsigned(2))
}

/// Shard 2 of a Nemotron-H model has no `general.architecture`, yet its
/// llama.cpp `ssm_a` still maps to the checkpoint's `A_log` with its shape,
/// in the whole catalog and in shard 2's share of it, which Metal loads that
/// shard's tensors from; building the shares reads each header once.
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
    let decay = catalog.get("backbone.layers.1.mixer.A_log");
    assert_eq!(decay.map(|tensor| tensor.shape.clone()), Some(vec![4]));
    assert!(catalog.get("backbone.layers.0.mixer.D").is_some());
    drop(super::super::header::reads::take());
    let mut shards = GgufModel::open(&paths[1])?.catalog_by_file()?;
    let reads = super::super::header::reads::take();
    std::fs::remove_dir_all(&directory)?;
    assert_eq!(reads, paths.iter().map(|path| (path.clone(), 1)).collect());
    let second = shards.remove(&paths[1]).unwrap_or_default();
    let decay = second.iter().find(|tensor| tensor.name == "backbone.layers.1.mixer.A_log");
    assert_eq!(
        decay.map(|tensor| (tensor.shape.clone(), &tensor.file)),
        Some((vec![4], &paths[1]))
    );
    assert!(second.iter().all(|tensor| tensor.name != "backbone.layers.0.mixer.D"));
    assert!(
        shards[&paths[0]]
            .iter()
            .any(|tensor| tensor.name == "backbone.layers.0.mixer.D")
    );
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
    let catalog = GgufModel::from_headers(shards.clone(), [])?.mapped_catalog()?;
    assert!(catalog.get("backbone.layers.1.mixer.A_log").is_some());
    let reversed = shards.into_iter().rev().collect::<Vec<_>>();
    assert!(GgufModel::from_headers(reversed, []).is_err());
    assert!(GgufModel::from_headers(Vec::new(), []).is_err());
    Ok(())
}

/// Gemma's `ffn_norm` is the norm before the feed-forward wherever its block's
/// `post_ffw_norm` is stored: in the same file, together in shard 2, or in
/// the other shard.
#[test]
fn sandwich_norms_map_alike_however_the_shards_split_them() -> Result<()> {
    let directory =
        std::env::temp_dir().join(format!("libmir-gguf-sandwich-{}", std::process::id()));
    std::fs::create_dir_all(&directory)?;
    let norms = ["blk.0.attn_norm.weight", "blk.1.ffn_norm.weight", "blk.1.post_ffw_norm.weight"];
    let mapped = |label: &str, split: &[&[&str]]| -> Result<Vec<String>> {
        let count = u32::try_from(split.len())?;
        let paths = (1..=count)
            .map(|number| directory.join(format!("{label}-{number:05}-of-{count:05}.gguf")))
            .collect::<Vec<_>>();
        for (index, (path, names)) in paths.iter().zip(split).enumerate() {
            let mut writer = Writer::new();
            if index == 0 {
                let total = u32::try_from(norms.len())?.to_le_bytes().to_vec();
                writer = writer.text("general.architecture", "gemma2")?;
                writer = writer.kv("split.tensors.count", 4, total);
            }
            let unsigned = |value: u32| value.to_le_bytes().to_vec();
            writer = writer.kv("split.no", 4, unsigned(u32::try_from(index)?)).kv(
                "split.count",
                4,
                unsigned(count),
            );
            for name in *names {
                writer = writer.tensor(name, &[4], GgmlType::F32)?;
            }
            std::fs::write(path, writer.bytes()?)?;
        }
        let catalog = GgufModel::open(&paths[0])?.mapped_catalog()?;
        let mut names = catalog.tensors.into_iter().map(|tensor| tensor.name).collect::<Vec<_>>();
        names.sort();
        Ok(names)
    };
    let one = mapped("one", &[&norms])?;
    let together = mapped("together", &[&norms[..1], &norms[1..]])?;
    let apart = mapped("apart", &[&norms[..2], &norms[2..]])?;
    let reversed = mapped("reversed", &[&[norms[0], norms[2]], &[norms[1]]])?;
    std::fs::remove_dir_all(&directory)?;
    assert!(
        one.contains(&"model.layers.1.pre_feedforward_layernorm.weight".to_owned()),
        "{one:?}"
    );
    assert_eq!(together, one);
    assert_eq!(apart, one);
    assert_eq!(reversed, one);
    Ok(())
}

/// Shard suffixes are read as ASCII bytes: accented and non-Latin stems stay
/// whole, malformed suffixes are no shard, and a numbered shard parses.
#[test]
fn shard_names_parse_without_cutting_characters() {
    let shard = |prefix, number, count| Some(ShardName { prefix, number, count });
    assert_eq!(ShardName::parse("model-aaaaé12345678"), None);
    assert_eq!(ShardName::parse("modèle-00002-of-00003"), shard("modèle", 2, 3));
    assert_eq!(ShardName::parse("模型-量化-00001-OF-00002"), shard("模型-量化", 1, 2));
    assert_eq!(ShardName::parse("-00001-of-00001"), shard("", 1, 1));
    assert_eq!(ShardName::parse("Qwen3-8B-Q4_K_M"), None);
    assert_eq!(ShardName::parse("m-0001-of-00002"), None);
    assert_eq!(ShardName::parse("m-00003-of-00002"), None);
    assert_eq!(ShardName::parse("m-00000-of-00002"), None);
    assert_eq!(ShardName::parse("m-0000a-of-00002"), None);
    assert_eq!(ShardName::parse("m-00001-on-00002"), None);
    assert_eq!(ShardName::parse("m_00001-of-00002"), None);
    assert_eq!(ShardName::parse("é-of-00002"), None);
}

/// An upper-case `.GGUF` extension still finds every numbered sibling,
/// spelled as the file given.
#[test]
fn upper_case_extensions_keep_their_sibling_shards() -> Result<()> {
    let directory = std::env::temp_dir().join(format!("libmir-gguf-upper-{}", std::process::id()));
    std::fs::create_dir_all(&directory)?;
    let paths = [1, 2].map(|number| directory.join(format!("weights-{number:05}-of-00002.GGUF")));
    for path in &paths {
        std::fs::write(path, b"")?;
    }
    let found = crate::gguf::shard_paths(&paths[1]);
    std::fs::remove_dir_all(&directory)?;
    assert_eq!(found?, paths.to_vec());
    Ok(())
}
