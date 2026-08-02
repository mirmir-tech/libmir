use std::collections::BTreeSet;

use serde::Deserialize;

use super::{
    BitsAndBytes4BitReference, Float8Reference, MxFp4Reference, MxFp8Reference, NvFp4Reference,
    TestResult, require,
};

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd)]
#[serde(rename_all = "snake_case")]
pub enum Family {
    Dense,
    DenseAndRouted,
    SharedRouted,
    ClampedRouted,
}

#[derive(Debug, Deserialize)]
pub struct Catalog {
    pub(super) schema: u32,
    pub fixtures: Vec<Fixture>,
}

#[derive(Debug, Deserialize)]
pub struct Fixture {
    pub family: Family,
    pub model_env: String,
    pub reference_env: String,
}

#[derive(Debug, Deserialize)]
pub struct Reference {
    pub(super) schema: u32,
    pub family: Family,
    pub model_type: Option<String>,
    pub dtypes: Vec<String>,
    #[serde(default)]
    pub affine: Option<AffineReference>,
    #[serde(default)]
    pub packed_int8: Option<PackedIntegerReference>,
    #[serde(default)]
    pub packed_int4: Option<PackedIntegerReference>,
    #[serde(default)]
    pub awq: Option<AwqReference>,
    #[serde(default)]
    pub gptq: Option<GptqReference>,
    #[serde(default)]
    pub float8: Option<Float8Reference>,
    #[serde(default)]
    pub mxfp4: Option<MxFp4Reference>,
    #[serde(default)]
    pub mxfp8: Option<MxFp8Reference>,
    #[serde(default)]
    pub nvfp4: Option<NvFp4Reference>,
    #[serde(default)]
    pub bitsandbytes_4bit: Option<BitsAndBytes4BitReference>,
    pub vocab_size: usize,
    pub context_len: usize,
    pub tokenizer: TokenizerReference,
    pub prompt_tokens: Vec<u32>,
    pub generated_tokens: Vec<u32>,
    pub first_logits: LogitsReference,
    pub metal: Option<ResourceGate>,
    pub cuda: Option<ResourceGate>,
    #[serde(default = "default_kv_cache_blocks")]
    pub kv_cache_blocks: u32,
}

#[derive(Debug, Deserialize)]
pub struct AffineReference {
    pub bits: Vec<u8>,
    pub group_sizes: Vec<usize>,
    pub parameter_dtype: String,
}

#[derive(Debug, Deserialize)]
pub struct PackedIntegerReference {
    pub bits: u8,
    pub scale_strategy: String,
    #[serde(default)]
    pub group_size: Option<usize>,
    pub signedness: String,
    pub zero_point: String,
    pub activation_order: String,
    pub packing: String,
    pub storage_dtype: String,
    pub scale_dtype: String,
}

#[derive(Debug, Deserialize)]
pub struct AwqReference {
    pub bits: u8,
    pub group_size: usize,
}

#[derive(Debug, Deserialize)]
pub struct GptqReference {
    pub bits: u8,
    pub group_size: usize,
    pub checkpoint_format: String,
    pub symmetric: bool,
    pub activation_order: bool,
    pub scale_dtype: String,
}

#[derive(Debug, Deserialize)]
pub struct TokenizerReference {
    pub vocabulary_entries: usize,
    pub max_token_id: u32,
    pub added_tokens: usize,
    pub stop_token_ids: Vec<u32>,
}

#[derive(Debug, Deserialize)]
pub struct LogitsReference {
    pub token_ids: Vec<u32>,
    pub scores: Vec<f32>,
    pub absolute_tolerance: f32,
    #[serde(default)]
    pub normalized: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceGate {
    pub max_load_active_bytes: u64,
    pub max_decode_active_bytes: u64,
    #[serde(default)]
    pub generated_tokens: Option<Vec<u32>>,
    #[serde(default)]
    pub first_logits: Option<LogitsReference>,
    #[serde(default)]
    pub generated_token_ties: Vec<GeneratedTokenTie>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeneratedTokenTie {
    pub position: usize,
    pub token_ids: Vec<u32>,
}

impl ResourceGate {
    pub(super) fn validate(&self) -> TestResult<()> {
        require(self.max_load_active_bytes > 0, "load memory gate must be positive")?;
        require(
            self.max_decode_active_bytes >= self.max_load_active_bytes,
            "decode memory gate must not be lower than the load gate",
        )?;
        for tie in &self.generated_token_ties {
            require(tie.token_ids.len() >= 2, "generation tie needs at least two token IDs")?;
            require(
                tie.token_ids.iter().collect::<BTreeSet<_>>().len() == tie.token_ids.len(),
                "generation tie token IDs must be unique",
            )?;
        }
        Ok(())
    }

    pub fn allows_generation(&self, actual: &[u32], expected: &[u32]) -> bool {
        actual.len() == expected.len()
            && actual.iter().zip(expected).enumerate().all(|(position, (actual, expected))| {
                actual == expected
                    || self.generated_token_ties.iter().any(|tie| {
                        tie.position == position
                            && tie.token_ids.contains(actual)
                            && tie.token_ids.contains(expected)
                    })
            })
    }
}

const fn default_kv_cache_blocks() -> u32 {
    128
}
