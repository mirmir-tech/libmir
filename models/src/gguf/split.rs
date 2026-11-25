//! Models split across `<name>-00001-of-0000N.gguf` files. The first shard
//! carries the model metadata; every shard records `split.no` and
//! `split.count`, and the first records `split.tensors.count`.

use std::path::{Path, PathBuf};

use super::{GgmlType, GgufHeader, GgufTensor, vectors::VectorSource};
use crate::{
    error::{ModelsError, Result},
    layout::WeightContainer,
    weights::{TensorCatalog, TensorInfo},
};

/// One GGUF model: a single file or every shard of a split model, in order.
#[derive(Debug, Clone)]
pub struct GgufModel {
    pub shards: Vec<(PathBuf, GgufHeader)>,
    pub(super) vectors: VectorSource,
}

impl GgufModel {
    /// Opens a single file, or all shards when `path` names any one of them.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let shards = shard_paths(path.as_ref())?
            .into_iter()
            .map(|shard| GgufHeader::read(&shard).map(|header| (shard, header)))
            .collect::<Result<Vec<_>>>()?;
        let model = Self { shards, vectors: VectorSource::Files };
        model.validate()?;
        Ok(model)
    }

    /// The header with the model metadata: the only file, or shard 1.
    #[must_use]
    pub fn header(&self) -> &GgufHeader {
        // `open` never builds an empty model.
        &self.shards[0].1
    }

    pub fn catalog(&self) -> Result<TensorCatalog> {
        let mut tensors: Vec<TensorInfo> = Vec::new();
        for (path, header) in &self.shards {
            tensors.extend(header.catalog(path)?.tensors);
        }
        let catalog = TensorCatalog::new(tensors);
        if let Some(pair) = catalog.tensors.windows(2).find(|pair| pair[0].name == pair[1].name) {
            return Err(ModelsError::InvalidGguf(format!(
                "tensor `{}` appears in two shards",
                pair[0].name
            )));
        }
        Ok(catalog)
    }

    /// The catalog under `SafeTensors` names across every shard: the names
    /// map by the model metadata, which only shard 1 carries, and checks
    /// that look at sibling tensors see the whole model.
    pub fn mapped_catalog(&self) -> Result<TensorCatalog> {
        let mut tensors = self.catalog()?.tensors;
        super::fused::split(self.header(), &mut tensors)?;
        super::names::map(self.header(), &mut tensors)?;
        Ok(TensorCatalog::new(tensors))
    }

    pub fn tensors(&self) -> impl Iterator<Item = &GgufTensor> {
        self.shards.iter().flat_map(|(_, header)| header.tensors.iter())
    }

    /// Payload bytes per GGML type across shards, largest first.
    #[must_use]
    pub fn type_bytes(&self) -> Vec<(GgmlType, u64)> {
        super::type_bytes(self.tensors())
    }

    pub(super) fn validate(&self) -> Result<()> {
        let count = self.shards.len();
        if count == 1 && self.header().unsigned("split.count").is_none_or(|declared| declared == 1)
        {
            return Ok(());
        }
        for (index, (path, header)) in self.shards.iter().enumerate() {
            let number = header.unsigned("split.no");
            let declared = header.unsigned("split.count");
            if number != Some(u64::try_from(index)?) || declared != Some(u64::try_from(count)?) {
                return Err(ModelsError::InvalidGguf(format!(
                    "{} declares shard {number:?} of {declared:?}; expected {index} of {count}",
                    path.display()
                )));
            }
        }
        let total = self.tensors().count();
        match self.header().unsigned("split.tensors.count") {
            Some(declared) if declared == u64::try_from(total)? => Ok(()),
            declared => Err(ModelsError::InvalidGguf(format!(
                "split model declares {declared:?} tensors; its shards hold {total}"
            ))),
        }
    }
}

/// The files of the model `path` belongs to: every shard of a split model in
/// order, or `path` alone.
pub fn shard_paths(path: &Path) -> Result<Vec<PathBuf>> {
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return Ok(vec![path.to_path_buf()]);
    };
    let Some(split) = name
        .strip_suffix(WeightContainer::Gguf.extension())
        .and_then(|stem| stem.strip_suffix('.'))
        .and_then(ShardName::parse)
    else {
        return Ok(vec![path.to_path_buf()]);
    };
    let directory = path.parent().unwrap_or_else(|| Path::new("."));
    (1..=split.count)
        .map(|number| {
            let shard = directory.join(split.renumbered(name, number));
            if shard.is_file() {
                Ok(shard)
            } else {
                Err(ModelsError::MissingFile(shard))
            }
        })
        .collect()
}

/// A gguf-split file stem, `<prefix>-00002-of-00003`: shard `number` of
/// `count`, numbered from 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShardName<'a> {
    pub prefix: &'a str,
    pub number: u32,
    pub count: u32,
}

impl<'a> ShardName<'a> {
    /// The suffix after the prefix: `-NNNNN-of-NNNNN`.
    const SUFFIX: usize = 15;

    /// Parses a file stem, without its extension. The suffix is checked as
    /// ASCII bytes before the stem is cut, so any UTF-8 prefix stays whole;
    /// `-of-` matches in any case.
    #[must_use]
    pub fn parse(stem: &'a str) -> Option<Self> {
        let start = stem.len().checked_sub(Self::SUFFIX)?;
        let suffix = &stem.as_bytes()[start..];
        let digits = |bytes: &[u8]| {
            bytes.iter().try_fold(0_u32, |value, &byte| {
                byte.is_ascii_digit().then(|| value * 10 + u32::from(byte - b'0'))
            })
        };
        if suffix[0] != b'-' || !suffix[6..10].eq_ignore_ascii_case(b"-of-") {
            return None;
        }
        let (number, count) = (digits(&suffix[1..6])?, digits(&suffix[10..])?);
        // `start` holds the ASCII `-`, so it is a character boundary.
        let prefix = stem.get(..start)?;
        (1..=count).contains(&number).then_some(Self { prefix, number, count })
    }

    /// `name`, the file this stem was parsed from, numbered `number` instead:
    /// its prefix, separator and extension are kept as they are spelled.
    fn renumbered(self, name: &str, number: u32) -> String {
        let head = self.prefix.len() + 1;
        format!("{}{number:05}{}", &name[..head], &name[head + 5..])
    }
}
