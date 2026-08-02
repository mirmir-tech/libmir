use mircuda::{DeviceBuffer, bf16};

use super::{
    MxFp4Bf16Linear, MxFp4CheckpointWeight, MxFp4EmbeddingLookup, MxFp4GatheredBf16Linear, buffer,
};
use crate::{Error, Result};

impl MxFp4Bf16Linear {
    pub fn execute(
        &self,
        input: &DeviceBuffer<bf16>,
        weight: &MxFp4CheckpointWeight,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        if self.spec.input_features != weight.input_features
            || self.spec.output_features != weight.output_features
            || self.has_bias != weight.bias.is_some()
        {
            return Err(Error::InvalidExecutionPlan(
                "MXFP4 plan and late-bound weight contract differ",
            ));
        }
        let scales = buffer(weight.scales.as_u8(), &weight.scales, "U8")?;
        let bias = weight
            .bias
            .as_ref()
            .map(|value| {
                value.as_bf16().ok_or_else(|| Error::DTypeMismatch {
                    name: value.name().into(),
                    expected: "BF16",
                })
            })
            .transpose()?;
        self.operation
            .execute(&self.stream, input, &weight.packed, scales, bias, output)
    }
}

impl MxFp4GatheredBf16Linear {
    pub fn execute(
        &self,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        weight: &MxFp4CheckpointWeight,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let matrices = match weight.layout {
            models::weights::BlockProjectionLayout::MatrixBank { matrices }
            | models::weights::BlockProjectionLayout::FusedGateUpBank {
                experts: matrices,
                interleaved: true,
            } => matrices,
            _ => 0,
        };
        if matrices != self.spec.matrices
            || self.spec.input_features != weight.input_features
            || self.spec.output_features != weight.output_features
            || self.has_bias != weight.bias.is_some()
        {
            return Err(Error::InvalidExecutionPlan(
                "gathered MXFP4 plan and weight contract differ",
            ));
        }
        let scales = buffer(weight.scales.as_u8(), &weight.scales, "U8")?;
        let bias = weight
            .bias
            .as_ref()
            .map(|value| {
                value.as_bf16().ok_or_else(|| Error::DTypeMismatch {
                    name: value.name().into(),
                    expected: "BF16",
                })
            })
            .transpose()?;
        self.operation.execute(
            &self.stream,
            &mut crate::kernels::MxFp4GatheredOperands {
                input,
                weight: &weight.packed,
                scales,
                bias,
                selected,
                output,
            },
        )
    }
}

impl MxFp4EmbeddingLookup {
    pub fn execute_batch(
        &self,
        selected: &DeviceBuffer<u32>,
        selected_start: usize,
        tokens: usize,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.operation.execute(
            &self.stream,
            &mut crate::kernels::MxFp4EmbeddingOperands {
                weight: &self.weight.packed,
                scales: buffer(self.weight.scales.as_u8(), &self.weight.scales, "U8")?,
                selected,
                output,
            },
            selected_start,
            tokens,
        )
    }

    pub fn validate_token(&self, token: u32) -> Result<()> {
        if usize::try_from(token)? < self.weight.output_features {
            Ok(())
        } else {
            Err(Error::InvalidToken {
                token,
                vocab: self.weight.output_features,
            })
        }
    }
}
