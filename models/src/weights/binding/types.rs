use serde::{Deserialize, Serialize};

use super::{
    AwqQuantization, BitsAndBytes4BitQuantization, BlockQuantization,
    CompressedIntegerQuantization, Float8Quantization, GptqQuantization, GroupedAffineQuantization,
    LogicalTensorRole,
};

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
        format: GroupedAffineQuantization,
        scales: String,
        biases: Option<String>,
        output_bias: Option<String>,
    },
    PackedInt8 {
        format: CompressedIntegerQuantization,
        scales: String,
        shape: String,
        zero_points: Option<String>,
        group_indices: Option<String>,
    },
    PackedInt4 {
        format: CompressedIntegerQuantization,
        scales: String,
        shape: String,
        zero_points: Option<String>,
        group_indices: Option<String>,
    },
    Awq {
        format: AwqQuantization,
        scales: String,
        zero_points: String,
    },
    Gptq {
        format: GptqQuantization,
        scales: String,
        zero_points: String,
        group_indices: String,
    },
    BitsAndBytes4Bit {
        format: BitsAndBytes4BitQuantization,
        absmax: String,
        quant_map: String,
        nested_absmax: Option<String>,
        nested_quant_map: Option<String>,
        quant_state: String,
        nested_offset_bits: Option<u32>,
    },
    Float8 {
        format: Float8Quantization,
        scale: Option<String>,
        input_scale: Option<String>,
        bias: Option<String>,
    },
    BlockQuantized {
        format: BlockQuantization,
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
            TensorStorage::PackedInt8 {
                scales,
                shape,
                zero_points,
                group_indices,
                ..
            }
            | TensorStorage::PackedInt4 {
                scales,
                shape,
                zero_points,
                group_indices,
                ..
            } => {
                sources.extend([scales.as_str(), shape.as_str()]);
                sources.extend(zero_points.as_deref());
                sources.extend(group_indices.as_deref());
            },
            TensorStorage::Awq { scales, zero_points, .. } => {
                sources.extend([scales.as_str(), zero_points.as_str()]);
            },
            TensorStorage::Gptq { scales, zero_points, group_indices, .. } => {
                sources.extend([scales.as_str(), zero_points.as_str(), group_indices.as_str()]);
            },
            TensorStorage::BitsAndBytes4Bit {
                absmax,
                quant_map,
                nested_absmax,
                nested_quant_map,
                quant_state,
                ..
            } => {
                sources.extend([absmax.as_str(), quant_map.as_str(), quant_state.as_str()]);
                sources.extend(nested_absmax.as_deref());
                sources.extend(nested_quant_map.as_deref());
            },
            TensorStorage::Float8 { scale, input_scale, bias, .. } => {
                sources.extend(scale.as_deref());
                sources.extend(input_scale.as_deref());
                sources.extend(bias.as_deref());
            },
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
