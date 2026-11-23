use serde::{Deserialize, Serialize};

use super::{
    AwqQuantization, BitsAndBytes4BitQuantization, BlockQuantization,
    CompressedIntegerQuantization, Float8Quantization, GgmlBlockQuantization, GptqQuantization,
    GroupedAffineQuantization, LogicalTensorRole,
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
    /// Compressed-tensors `pack-quantized` integers; `format.bits` is the
    /// width.
    PackedInteger {
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
    /// GGML blocks from a GGUF file; scales live inside the blocks.
    Ggml {
        format: GgmlBlockQuantization,
        /// A dense output bias beside the blocks (gpt-oss).
        bias: Option<String>,
        /// An F32 output scale: one value, or one per expert of a bank
        /// (llama.cpp's `.scale`, e.g. `ModelOpt` NVFP4's `weight_scale_2`).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        scale: Option<String>,
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
    /// The physical tensors runtimes upload in their checkpoint dtype rather
    /// than as BF16: a quantized weight's companions (scales, zeros, FP8
    /// `weight_scale` and `input_scale`), a GGML output scale, and a sigmoid
    /// router's correction bias, which decides the expert selection.
    #[must_use]
    pub fn full_precision_sources(&self) -> Vec<&str> {
        if matches!(
            self.role.layer_tensor(),
            Some(super::roles::LayerTensorRole::RouterCorrectionBias)
        ) {
            return vec![self.source.as_str()];
        }
        match &self.storage {
            TensorStorage::AffineQuantized { .. }
            | TensorStorage::BlockQuantized { .. }
            | TensorStorage::Float8 { .. }
            | TensorStorage::PackedInteger { .. }
            | TensorStorage::Awq { .. }
            | TensorStorage::Gptq { .. }
            | TensorStorage::BitsAndBytes4Bit { .. } => self.physical_sources(),
            _ => self.output_scale().into_iter().collect(),
        }
    }

    /// The F32 output scale of a GGML projection (llama.cpp's `.scale`),
    /// which runtimes keep in F32.
    #[must_use]
    pub fn output_scale(&self) -> Option<&str> {
        match &self.storage {
            TensorStorage::Ggml { scale, .. } => scale.as_deref(),
            _ => None,
        }
    }

    /// The dense companion bias of a dense tensor (a `LayerNorm`'s `bias`).
    #[must_use]
    pub fn dense_bias(&self) -> Option<&str> {
        match &self.storage {
            TensorStorage::Dense { bias, .. } => bias.as_deref(),
            _ => None,
        }
    }

    #[must_use]
    pub fn physical_sources(&self) -> Vec<&str> {
        let mut sources = vec![self.source.as_str()];
        match &self.storage {
            TensorStorage::Dense { bias, .. } => sources.extend(bias.as_deref()),
            TensorStorage::Ggml { bias, scale, .. } => {
                sources.extend(bias.as_deref());
                sources.extend(scale.as_deref());
            },
            TensorStorage::AffineQuantized { scales, biases, output_bias, .. } => {
                sources.push(scales);
                sources.extend(biases.as_deref());
                sources.extend(output_bias.as_deref());
            },
            TensorStorage::PackedInteger {
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
