use super::*;
use crate::error::ModelsError;

mod cohere;
mod container;
mod fused;
mod gemma4;
mod glm;
mod granite;
mod hybrid;
mod import;
mod latent;
mod scales;
mod split;
mod vision;

pub type TestResult<T = ()> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Writes a minimal GGUF v3 file in memory, the way llama.cpp lays it out.
struct Writer {
    metadata: Vec<(String, Vec<u8>)>,
    tensors: Vec<Entry>,
}

struct Entry {
    name: String,
    dims: Vec<u64>,
    kind: u32,
    offset: u64,
    bytes: u64,
}

const ALIGNMENT: u64 = 32;

fn length(value: usize) -> Result<u64> {
    Ok(u64::try_from(value)?)
}

impl Writer {
    const fn new() -> Self {
        Self {
            metadata: Vec::new(),
            tensors: Vec::new(),
        }
    }

    fn string(value: &str) -> Result<Vec<u8>> {
        let mut bytes = length(value.len())?.to_le_bytes().to_vec();
        bytes.extend_from_slice(value.as_bytes());
        Ok(bytes)
    }

    fn kv(mut self, key: &str, kind: u32, payload: Vec<u8>) -> Self {
        let mut value = kind.to_le_bytes().to_vec();
        value.extend(payload);
        self.metadata.push((key.to_owned(), value));
        self
    }

    fn text(self, key: &str, value: &str) -> Result<Self> {
        let payload = Self::string(value)?;
        Ok(self.kv(key, 8, payload))
    }

    /// A tensor of whole blocks, placed after the previous one.
    fn tensor(mut self, name: &str, dims: &[u64], kind: GgmlType) -> Result<Self> {
        let bytes = kind.row_bytes(dims[0])? * dims[1..].iter().product::<u64>();
        let offset = self.end().div_ceil(ALIGNMENT) * ALIGNMENT;
        self.tensors.push(Entry {
            name: name.to_owned(),
            dims: dims.to_vec(),
            kind: kind.id(),
            offset,
            bytes,
        });
        Ok(self)
    }

    fn end(&self) -> u64 {
        self.tensors.last().map_or(0, |last| last.offset + last.bytes)
    }

    fn bytes(&self) -> Result<Vec<u8>> {
        let mut out = MAGIC.to_vec();
        out.extend(3_u32.to_le_bytes());
        out.extend(length(self.tensors.len())?.to_le_bytes());
        out.extend(length(self.metadata.len())?.to_le_bytes());
        for (key, value) in &self.metadata {
            out.extend(Self::string(key)?);
            out.extend(value);
        }
        for entry in &self.tensors {
            out.extend(Self::string(&entry.name)?);
            out.extend(u32::try_from(entry.dims.len())?.to_le_bytes());
            for dim in &entry.dims {
                out.extend(dim.to_le_bytes());
            }
            out.extend(entry.kind.to_le_bytes());
            out.extend(entry.offset.to_le_bytes());
        }
        let header = length(out.len())?.div_ceil(ALIGNMENT) * ALIGNMENT;
        out.resize(usize::try_from(header + self.end())?, 0);
        Ok(out)
    }
}

fn parse(bytes: &[u8]) -> std::result::Result<GgufHeader, ParseError> {
    GgufHeader::parse(bytes, u64::try_from(bytes.len()).ok())
}

const fn invalid(result: &std::result::Result<GgufHeader, ParseError>) -> bool {
    matches!(result, Err(ParseError::Invalid(_)))
}

fn qwen_like() -> Result<Writer> {
    Writer::new()
        .text("general.architecture", "qwen3")?
        .kv("qwen3.block_count", 4, 2_u32.to_le_bytes().to_vec())
        .tensor("token_embd.weight", &[256, 8], GgmlType::Q4_K)?
        .tensor("blk.0.attn_q.weight", &[64, 64], GgmlType::Q8_0)?
        .tensor("output_norm.weight", &[256], GgmlType::F32)
}

#[test]
fn block_sizes_match_ggml() {
    let expected = [
        (GgmlType::Q4_K, 256, 144),
        (GgmlType::Q6_K, 256, 210),
        (GgmlType::Q8_0, 32, 34),
        (GgmlType::IQ4_XS, 256, 136),
        (GgmlType::MXFP4, 32, 17),
        (GgmlType::NVFP4, 64, 36),
        (GgmlType::Q1_0, 128, 18),
    ];
    for (kind, elements, bytes) in expected {
        assert_eq!((kind.block_elements(), kind.block_bytes()), (elements, bytes), "{kind}");
    }
    assert!((GgmlType::Q4_K.bits_per_weight() - 4.5).abs() < f64::EPSILON);
}

#[test]
fn every_type_has_one_row_and_unknown_ids_are_named() -> Result<()> {
    for kind in GgmlType::all() {
        assert_eq!(GgmlType::from_id(kind.id())?, kind);
    }
    assert_eq!(GgmlType::all().count(), 36);
    for removed in [4, 5, 31, 36, 44] {
        assert!(
            matches!(GgmlType::from_id(removed), Err(ModelsError::UnsupportedGgmlType(id)) if id == removed)
        );
    }
    Ok(())
}

#[test]
fn parses_metadata_tensors_and_catalog() -> TestResult {
    let bytes = qwen_like()?.bytes()?;
    let header = parse(&bytes).map_err(|error| format!("{error:?}"))?;
    assert_eq!(header.version, 3);
    assert_eq!(header.architecture(), Some("qwen3"));
    assert_eq!(header.unsigned("qwen3.block_count"), Some(2));
    assert_eq!(header.data_start % ALIGNMENT, 0);
    let catalog = header.catalog(Path::new("model.gguf"))?;
    let embedding = catalog.get("token_embd.weight").ok_or("embedding missing")?;
    assert_eq!(embedding.dtype, "Q4_K");
    assert_eq!(embedding.shape, [8, 256]);
    assert_eq!(embedding.payload_bytes()?, 8 * 144);
    assert_eq!(header.type_bytes().first().map(|entry| entry.0), Some(GgmlType::Q8_0));
    Ok(())
}

#[test]
fn a_short_prefix_asks_for_more_bytes() -> Result<()> {
    let bytes = qwen_like()?.bytes()?;
    // The header proper ends before byte 260; the rest up to 288 is padding.
    for cut in [3_usize, 20, 60, 120, 200, 250] {
        let result = GgufHeader::parse(&bytes[..cut], u64::try_from(bytes.len()).ok());
        let needed = length(cut)?;
        assert!(
            matches!(result, Err(ParseError::Truncated { needed: more }) if more > needed),
            "cut at {cut}"
        );
    }
    Ok(())
}

#[test]
fn rejects_malformed_headers() -> Result<()> {
    let good = qwen_like()?.bytes()?;
    let mut magic = good.clone();
    magic[0] = b'X';
    assert!(invalid(&parse(&magic)));
    let mut version = good.clone();
    version[4] = 1;
    assert!(invalid(&parse(&version)));
    let mut count = good;
    count[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(invalid(&parse(&count)));
    // A Q4_K row of 96 values is not whole 256-value blocks: written as Q8_0
    // (three blocks), then relabelled Q4_K.
    let mut rows = Writer::new().tensor("bad", &[96, 2], GgmlType::Q8_0)?;
    rows.tensors[0].kind = GgmlType::Q4_K.id();
    assert!(invalid(&parse(&rows.bytes()?)));
    let mut unknown = Writer::new().tensor("odd", &[32], GgmlType::Q8_0)?;
    unknown.tensors[0].kind = 31;
    assert!(matches!(
        parse(&unknown.bytes()?),
        Err(ParseError::Invalid(ModelsError::UnsupportedGgmlType(31)))
    ));
    Ok(())
}

#[test]
fn rejects_truncated_payloads_and_duplicates() -> Result<()> {
    let bytes = qwen_like()?.bytes()?;
    let short = &bytes[..bytes.len() - 1];
    assert!(invalid(&parse(short)));
    let twice = Writer::new().text("general.name", "a")?.text("general.name", "b")?.bytes()?;
    assert!(invalid(&parse(&twice)));
    let tensors = Writer::new()
        .tensor("w", &[32], GgmlType::Q8_0)?
        .tensor("w", &[32], GgmlType::Q8_0)?
        .bytes()?;
    assert!(invalid(&parse(&tensors)));
    Ok(())
}

#[test]
fn recognizes_split_names() -> Result<()> {
    let shards = shard_paths(Path::new("/nonexistent/model-00002-of-00003.gguf"));
    assert!(matches!(shards, Err(ModelsError::MissingFile(_))));
    assert_eq!(shard_paths(Path::new("model.gguf"))?, [PathBuf::from("model.gguf")]);
    assert_eq!(shard_paths(Path::new("m-1-of-2.gguf"))?, [PathBuf::from("m-1-of-2.gguf")]);
    Ok(())
}

/// `LIBMIR_GGUF=/path/model.gguf cargo test -p libmir-models -- --ignored
/// gguf`; `examples/gguf-inspect.rs` prints the same inventory.
#[test]
#[ignore = "reads a real GGUF file named by LIBMIR_GGUF"]
fn inspects_a_real_file() -> TestResult {
    let path = std::env::var("LIBMIR_GGUF")?;
    let model = GgufModel::open(&path)?;
    assert!(model.header().architecture().is_some());
    assert_eq!(model.catalog()?.len(), model.tensors().count());
    Ok(())
}
