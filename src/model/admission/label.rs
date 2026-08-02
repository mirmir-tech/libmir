use super::WeightEncoding;

impl WeightEncoding {
    #[must_use]
    /// Returns a compact human-readable encoding label.
    pub fn label(&self) -> String {
        match self {
            Self::Dense { dtype } => format!("Dense {dtype}"),
            Self::Affine { format } => format!(
                "Grouped affine {} G{} {:?}/{:?}",
                format.bits.get(),
                format.group_size,
                format.scale_dtype,
                format.zero_point
            ),
            Self::PackedInt8 { format } | Self::PackedInt4 { format } => format!(
                "Compressed INT{} {:?}/{:?}",
                format.bits.get(),
                format.scale_strategy,
                format.signedness
            ),
            Self::Awq { format } => format!(
                "AWQ GEMM {} G{} {:?}",
                format.bits.get(),
                format.group_size,
                format.packing
            ),
            Self::Gptq { format } => format!(
                "GPTQ {} G{} {:?}/{:?}",
                format.bits.get(),
                format.group_size,
                format.checkpoint_format,
                format.packing
            ),
            Self::BitsAndBytes4Bit { format } => format!(
                "bitsandbytes {} B{}{}",
                format.quant_type.as_str().to_ascii_uppercase(),
                format.block_size,
                format
                    .nested_block_size
                    .map_or(String::new(), |block| format!(" nested B{block}"))
            ),
            Self::Float8 { format } => format!(
                "FP8 {:?} {:?}/{:?}",
                format.format, format.scale_mode, format.scale_granularity
            ),
            Self::MxFp4 { format } => format!("MXFP4 B{}", format.block_size),
            Self::MxFp8 { format } => format!("MXFP8 B{}", format.block_size),
            Self::NvFp4 { format } => format!("NVFP4 B{}", format.block_size),
        }
    }
}
