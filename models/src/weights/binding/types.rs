use serde::{Deserialize, Serialize};

use super::LogicalTensorRole;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WeightBindingPlan {
    pub tensors: Vec<TensorBinding>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorBinding {
    pub role: LogicalTensorRole,
    pub source: String,
    pub shape: Vec<usize>,
    pub logical_shape: Option<Vec<usize>>,
    pub transforms: Vec<BindingTransform>,
    pub storage: TensorStorage,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BindingTransform {
    Transpose,
    FusedQkv { query: usize, key: usize, value: usize },
    FusedGateUp { interleaved: bool },
    StackedExperts { count: usize },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TensorStorage {
    Dense {
        dtype: String,
        bias: Option<String>,
    },
    AffineQuantized {
        dtype: String,
        bits: Option<u8>,
        scales: String,
        biases: Option<String>,
        output_bias: Option<String>,
        group_size: Option<usize>,
    },
    PackedInt8 {
        dtype: String,
        scales: String,
    },
    BlockQuantized {
        format: BlockFormat,
        scales: String,
        global_scale: Option<String>,
        input_scale: Option<String>,
        bias: Option<String>,
        packing: TensorPacking,
    },
    Auxiliary {
        dtype: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlockFormat {
    MxFp4,
    NvFp4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorPacking {
    Separate,
    InterleavedGateUp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertProjectionLayout {
    InterleavedGateUp,
    SeparateGateUp,
}

impl TensorBinding {
    #[must_use]
    pub fn physical_sources(&self) -> Vec<&str> {
        let mut sources = vec![self.source.as_str()];
        match &self.storage {
            TensorStorage::Dense { bias, .. } => sources.extend(bias.as_deref()),
            TensorStorage::AffineQuantized { scales, biases, output_bias, .. } => {
                sources.push(scales);
                sources.extend(biases.as_deref());
                sources.extend(output_bias.as_deref());
            },
            TensorStorage::PackedInt8 { scales, .. } => sources.push(scales),
            TensorStorage::BlockQuantized {
                scales, global_scale, input_scale, bias, ..
            } => {
                sources.push(scales);
                sources.extend(global_scale.as_deref());
                sources.extend(input_scale.as_deref());
                sources.extend(bias.as_deref());
            },
            TensorStorage::Auxiliary { .. } => {},
        }
        sources
    }
}
