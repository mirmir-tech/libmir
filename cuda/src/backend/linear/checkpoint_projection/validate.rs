use super::{super::validate_dense, CheckpointProjectionWeight};
use crate::{CudaTensor, Error, Result};

impl CheckpointProjectionWeight {
    /// The projection format this weight's storage executes as.
    #[must_use]
    pub(in crate::backend) const fn format(&self) -> crate::ProjectionFormat {
        use crate::ProjectionFormat;
        match self {
            Self::Affine(_) => ProjectionFormat::Affine,
            Self::Dense(_) => ProjectionFormat::Bf16,
            Self::DirectFp8(_) => ProjectionFormat::DirectFp8,
            Self::MxFp4(_) => ProjectionFormat::MxFp4,
            Self::Ggml(_) => ProjectionFormat::Ggml,
            Self::MxFp8(_) => ProjectionFormat::MxFp8,
            Self::NvFp4(_) | Self::NvFp4WeightOnly(_) => ProjectionFormat::NvFp4,
            Self::PackedInteger(_) => ProjectionFormat::PackedInteger,
        }
    }

    pub(in crate::backend) fn dense_bf16(&self) -> Option<&CudaTensor> {
        match self {
            Self::Dense(weight) if weight.dtype() == crate::CudaTensorDType::Bf16 => Some(weight),
            _ => None,
        }
    }

    pub(in crate::backend) fn affine_format(
        &self,
        matrices: usize,
        input: usize,
        output: usize,
    ) -> Result<Option<(usize, usize)>> {
        match self {
            Self::Affine(weight) => {
                let config = weight.infer_config(matrices, input, output)?;
                Ok(Some((config.group_size, config.bits)))
            },
            Self::Dense(weight) => {
                validate_dense(weight, matrices, input, output)?;
                Ok(None)
            },
            Self::DirectFp8(weight) => {
                weight.validate(input, output)?;
                Ok(None)
            },
            Self::MxFp4(weight) => {
                if matrices != 1 {
                    return Err(Error::InvalidExecutionPlan(
                        "MXFP4 checkpoint projection does not support matrix banks",
                    ));
                }
                weight.validate(input, output)?;
                Ok(None)
            },
            Self::Ggml(weight) => {
                if matrices != 1 {
                    return Err(Error::InvalidExecutionPlan(
                        "GGML checkpoint projection does not support matrix banks",
                    ));
                }
                weight.validate(input, output)?;
                Ok(None)
            },
            Self::MxFp8(weight) => {
                if matrices != 1 {
                    return Err(Error::InvalidExecutionPlan(
                        "MXFP8 checkpoint projection does not support matrix banks",
                    ));
                }
                weight.validate(input, output)?;
                Ok(None)
            },
            Self::NvFp4(weight) => {
                if matrices != 1 || weight.config() != crate::NvFp4Config::new(input, output) {
                    return Err(Error::InvalidExecutionPlan(
                        "NVFP4 checkpoint projection geometry does not match",
                    ));
                }
                Ok(None)
            },
            Self::NvFp4WeightOnly(weight) => {
                if matrices != 1 || weight.config() != crate::NvFp4Config::new(input, output) {
                    return Err(Error::InvalidExecutionPlan(
                        "NVFP4 weight-only projection geometry does not match",
                    ));
                }
                Ok(None)
            },
            Self::PackedInteger(weight) => {
                if matrices != 1 {
                    return Err(Error::InvalidQuantizedGemv(
                        "packed integer checkpoint projection does not support matrix banks",
                    ));
                }
                weight.validate(input, output)?;
                Ok(None)
            },
        }
    }

    pub(in crate::backend) fn validate(
        &self,
        matrices: usize,
        input: usize,
        output: usize,
        group_size: usize,
        bits: usize,
    ) -> Result<()> {
        match self {
            Self::Affine(weight) => weight.validate(matrices, input, output, group_size, bits),
            Self::Dense(weight) => validate_dense(weight, matrices, input, output),
            Self::DirectFp8(weight) if matrices == 1 => weight.validate(input, output),
            Self::DirectFp8(_) => Err(Error::InvalidExecutionPlan(
                "direct FP8 checkpoint projection does not support matrix banks",
            )),
            Self::MxFp4(weight) if matrices == 1 => weight.validate(input, output),
            Self::MxFp4(_) => Err(Error::InvalidExecutionPlan(
                "MXFP4 checkpoint projection does not support matrix banks",
            )),
            Self::Ggml(weight) if matrices == 1 => weight.validate(input, output),
            Self::Ggml(_) => Err(Error::InvalidExecutionPlan(
                "GGML checkpoint projection does not support matrix banks",
            )),
            Self::MxFp8(weight) if matrices == 1 => weight.validate(input, output),
            Self::MxFp8(_) => Err(Error::InvalidExecutionPlan(
                "MXFP8 checkpoint projection does not support matrix banks",
            )),
            Self::NvFp4(weight)
                if matrices == 1 && weight.config() == crate::NvFp4Config::new(input, output) =>
            {
                Ok(())
            },
            Self::NvFp4(_) => Err(Error::InvalidExecutionPlan(
                "NVFP4 checkpoint projection geometry does not match",
            )),
            Self::NvFp4WeightOnly(weight)
                if matrices == 1 && weight.config() == crate::NvFp4Config::new(input, output) =>
            {
                Ok(())
            },
            Self::NvFp4WeightOnly(_) => Err(Error::InvalidExecutionPlan(
                "NVFP4 weight-only projection geometry does not match",
            )),
            Self::PackedInteger(weight) if matrices == 1 => weight.validate(input, output),
            Self::PackedInteger(_) => Err(Error::InvalidQuantizedGemv(
                "packed integer checkpoint projection does not support matrix banks",
            )),
        }
    }
}
