use foundation::model::BackendTarget;
use models::{
    execution::ArchitectureRequirements,
    weights::{
        AwqQuantization, BitsAndBytes4BitQuantization, BlockFormat, BlockQuantization,
        CompressedIntegerQuantization, EncoderBindingPlan, Float8Quantization, GptqQuantization,
        GroupedAffineQuantization, TensorStorage, WeightBindingPlan,
    },
};
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
/// Conservative result of one checkpoint/backend admission check.
pub enum AdmissionStatus {
    /// Every required execution contract has been validated.
    Supported,
    /// The format is recognized but execution coverage is incomplete.
    Partial,
    /// At least one required format or operation is unavailable.
    Unsupported,
    /// Available metadata is insufficient to decide.
    Unknown,
}

impl AdmissionStatus {
    #[must_use]
    /// Returns the stable protocol spelling of this status.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Supported => "supported",
            Self::Partial => "partial",
            Self::Unsupported => "unsupported",
            Self::Unknown => "unknown",
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Stable category of a backend admission check.
pub enum AdmissionCheckKind {
    /// Dense floating-point weights.
    Dense,
    /// Grouped affine integer weights.
    Affine,
    /// Packed compressed-tensors INT8 weights.
    PackedInt8,
    /// Packed compressed-tensors INT4 weights.
    PackedInt4,
    /// `AutoAWQ` GEMM packed weights.
    Awq,
    /// GPTQ input-packed weights.
    Gptq,
    /// bitsandbytes NF4 or FP4 block weights.
    BitsAndBytes4Bit,
    /// Direct FP8 floating-point weights.
    Float8,
    /// OCP MXFP4 block weights.
    MxFp4,
    /// OCP MXFP8 block weights.
    MxFp8,
    /// NVIDIA NVFP4 block weights.
    NvFp4,
    /// Semantic model composition and backend lowering.
    Architecture,
    /// Vision tensors and image preprocessing contract.
    Vision,
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// One independently explainable backend admission decision.
pub struct AdmissionCheck {
    /// Capability family inspected by this check.
    pub kind: AdmissionCheckKind,
    /// Result of this check.
    pub status: AdmissionStatus,
    /// User-facing explanation or missing requirement.
    pub detail: String,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
/// Physical representation of one family of checkpoint weights.
pub enum WeightEncoding {
    /// Ordinary floating-point tensor storage.
    Dense {
        /// Safetensors dtype name.
        dtype: String,
    },
    /// Packed grouped affine integer storage.
    Affine {
        /// Complete physical quantization contract.
        format: GroupedAffineQuantization,
    },
    /// Compressed-tensors packed INT8 storage.
    PackedInt8 {
        /// Complete physical integer packing and quantization contract.
        format: CompressedIntegerQuantization,
    },
    /// Compressed-tensors grouped INT4 storage.
    PackedInt4 {
        /// Complete physical integer packing and quantization contract.
        format: CompressedIntegerQuantization,
    },
    /// `AutoAWQ` GEMM W4A16 storage.
    Awq {
        /// Complete asymmetric packing contract.
        format: AwqQuantization,
    },
    /// GPTQ input-packed storage.
    Gptq {
        /// Complete packing and quantizer contract.
        format: GptqQuantization,
    },
    /// bitsandbytes NF4 or FP4 storage.
    BitsAndBytes4Bit {
        /// Complete block, dtype, and nested-scale contract.
        format: BitsAndBytes4BitQuantization,
    },
    /// Direct FP8 projection storage.
    Float8 {
        /// Complete value and scale contract.
        format: Float8Quantization,
    },
    /// OCP MXFP4 blocks with shared scales.
    MxFp4 {
        /// Complete block and scale hierarchy.
        format: BlockQuantization,
    },
    /// OCP MXFP8 blocks with shared exponent scales.
    MxFp8 {
        /// Complete block and scale hierarchy.
        format: BlockQuantization,
    },
    /// NVIDIA NVFP4 blocks with local and global scales.
    NvFp4 {
        /// Complete block and scale hierarchy.
        format: BlockQuantization,
    },
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
/// Deduplicated physical encodings required by a checkpoint contract.
pub struct CheckpointEncoding {
    /// Distinct weight representations found in semantic bindings.
    pub weights: Vec<WeightEncoding>,
}

impl CheckpointEncoding {
    #[must_use]
    /// Derives physical encodings from validated semantic weight bindings.
    pub fn from_bindings(bindings: &WeightBindingPlan) -> Self {
        Self::from_storage(bindings.tensors.iter().map(|binding| &binding.storage))
    }

    /// Derives physical encodings from validated encoder bindings.
    #[must_use]
    pub fn from_encoder_bindings(bindings: &EncoderBindingPlan) -> Self {
        Self::from_storage(bindings.tensors.iter().map(|binding| &binding.storage))
    }

    pub(crate) fn include_dense_dtypes(&mut self, dtypes: &[String]) {
        self.weights
            .extend(dtypes.iter().cloned().map(|dtype| WeightEncoding::Dense { dtype }));
        self.weights.sort();
        self.weights.dedup();
    }

    fn from_storage<'a>(storage: impl Iterator<Item = &'a TensorStorage>) -> Self {
        let mut weights = storage
            .filter_map(|storage| match storage {
                TensorStorage::Dense { dtype, .. } => {
                    Some(WeightEncoding::Dense { dtype: dtype.clone() })
                },
                TensorStorage::AffineQuantized { format, .. } => {
                    Some(WeightEncoding::Affine { format: *format })
                },
                TensorStorage::PackedInt8 { format, .. } => {
                    Some(WeightEncoding::PackedInt8 { format: *format })
                },
                TensorStorage::PackedInt4 { format, .. } => {
                    Some(WeightEncoding::PackedInt4 { format: *format })
                },
                TensorStorage::Awq { format, .. } => Some(WeightEncoding::Awq { format: *format }),
                TensorStorage::Gptq { format, .. } => {
                    Some(WeightEncoding::Gptq { format: *format })
                },
                TensorStorage::BitsAndBytes4Bit { format, .. } => {
                    Some(WeightEncoding::BitsAndBytes4Bit { format: *format })
                },
                TensorStorage::Float8 { format, .. } => {
                    Some(WeightEncoding::Float8 { format: *format })
                },
                TensorStorage::BlockQuantized { format, .. } => Some(match format.format {
                    BlockFormat::MxFp4 => WeightEncoding::MxFp4 { format: *format },
                    BlockFormat::MxFp8 => WeightEncoding::MxFp8 { format: *format },
                    BlockFormat::NvFp4 => WeightEncoding::NvFp4 { format: *format },
                }),
                TensorStorage::Auxiliary { .. } => None,
            })
            .collect::<Vec<_>>();
        weights.sort();
        weights.dedup();
        Self { weights }
    }

    #[must_use]
    /// Returns a compact label covering every distinct weight representation.
    pub fn label(&self) -> String {
        if self.weights.is_empty() {
            "Unknown".into()
        } else {
            self.weights.iter().map(WeightEncoding::label).collect::<Vec<_>>().join(" + ")
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// Static checkpoint compatibility report for one accelerator backend.
pub struct BackendAdmissionReport {
    /// Accelerator backend evaluated by this report.
    pub backend: BackendTarget,
    /// Aggregate result across every check.
    pub status: AdmissionStatus,
    /// Physical checkpoint representations that were evaluated.
    pub encoding: CheckpointEncoding,
    /// Normalized task and decoder capabilities required by the model.
    pub architecture: Option<ArchitectureRequirements>,
    /// Individual decisions and actionable explanations.
    pub checks: Vec<AdmissionCheck>,
}
