use mircuda::{DeviceBuffer, MxFp8EmbeddingOperands, MxFp8GatheredOperands, bf16};

use super::{
    MxFp8Bf16Linear, MxFp8CheckpointWeight, MxFp8EmbeddingLookup, MxFp8GatheredBf16Linear, dtype,
};
use crate::{Error, Result};

impl MxFp8Bf16Linear {
    /// Enqueues one MXFP8 projection without allocation or synchronization.
    pub fn execute(
        &self,
        input: &DeviceBuffer<bf16>,
        weight: &MxFp8CheckpointWeight,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        if self.spec.input_features() != weight.input_features
            || self.spec.output_features() != weight.output_features
        {
            return Err(Error::InvalidExecutionPlan(
                "MXFP8 plan and late-bound weight contract differ",
            ));
        }
        self.operation.execute(&self.stream, &self.pool, input, weight, output)
    }
}

impl MxFp8GatheredBf16Linear {
    pub fn execute(
        &self,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        weight: &MxFp8CheckpointWeight,
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
        if matrices != self.matrices
            || weight.input_features != self.input_features
            || weight.output_features != self.output_features
            || weight.bias.is_some() != self.has_bias
        {
            return Err(Error::InvalidExecutionPlan(
                "gathered MXFP8 plan and weight contract differ",
            ));
        }
        Ok(self.operation.execute(
            &self.stream,
            &mut MxFp8GatheredOperands {
                input,
                weight: weight.weight.as_u32().ok_or_else(|| dtype(&weight.weight, "U32"))?,
                scales: weight.scales.as_u8().ok_or_else(|| dtype(&weight.scales, "U8"))?,
                bias: weight
                    .bias
                    .as_ref()
                    .map(|value| value.as_bf16().ok_or_else(|| dtype(value, "BF16")))
                    .transpose()?,
                selected,
                output,
            },
        )?)
    }
}

impl MxFp8EmbeddingLookup {
    /// Enqueues selected checkpoint rows as BF16 embeddings.
    pub fn execute_batch(
        &self,
        selected: &DeviceBuffer<u32>,
        selected_start: usize,
        tokens: usize,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.operation.execute(
            &self.stream,
            MxFp8EmbeddingOperands {
                weight: self
                    .weight
                    .weight
                    .as_u32()
                    .ok_or_else(|| dtype(&self.weight.weight, "U32"))?,
                scales: self
                    .weight
                    .scales
                    .as_u8()
                    .ok_or_else(|| dtype(&self.weight.scales, "U8"))?,
                selected,
                output,
            },
            selected_start,
            tokens,
        )?;
        Ok(())
    }

    /// Validates one token identifier before device execution.
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
