use mircuda::{DeviceBuffer, bf16};
use runtime::kv::{BlockTable, KvWritePlan};

use super::{LayerPrefill, PrefillSignature, SessionLayer, SharedLayerPrefill};
use crate::{Error, PagedPrefillBatch, Result};

impl SessionLayer {
    pub(in crate::backend::model) const fn prefill_signature(&self) -> PrefillSignature {
        match self {
            Self::Moe(layer) => {
                let mut config = layer.template.config();
                if layer.template.experts_are_dense() {
                    config.attention.layer = 0;
                }
                PrefillSignature::Moe(config)
            },
            Self::Dense(layer) => {
                let mut config = layer.template.config();
                if !matches!(config.attention.projection_format, crate::ProjectionFormat::NvFp4) {
                    config.attention.layer = 0;
                }
                PrefillSignature::Dense(config)
            },
        }
    }

    pub(in crate::backend::model) fn instantiate_shared_prefill(
        &self,
        tokens: usize,
    ) -> Result<SharedLayerPrefill> {
        let signature = self.prefill_signature();
        let plan = match self {
            Self::Moe(layer) => layer.template.instantiate_prefill(tokens)?.into(),
            Self::Dense(layer) => layer.template.instantiate_prefill(tokens)?.into(),
        };
        Ok(SharedLayerPrefill::new(signature, plan))
    }

    pub(in crate::backend::model) fn execute_shared_prefill_batch(
        &mut self,
        prefill: LayerPrefill<'_>,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        batch: &PagedPrefillBatch,
    ) -> Result<()> {
        match (self, prefill) {
            (Self::Moe(layer), LayerPrefill::Moe(prefill)) => {
                let decode = layer.decode.as_mut().ok_or(Error::InvalidDecoderKernel(
                    "CUDA prefill executor belongs to model graph",
                ))?;
                decode.execute_prefill_batch(prefill, input, batch, output)
            },
            (Self::Dense(layer), LayerPrefill::Dense(prefill)) => {
                let weights = layer.template.weights();
                let decode = layer.decode.as_mut().ok_or(Error::InvalidDecoderKernel(
                    "CUDA prefill executor belongs to model graph",
                ))?;
                prefill.execute_batch(decode, input, weights, batch, output)
            },
            _ => Err(Error::InvalidDecoderKernel(
                "shared CUDA prefill plan differs from decoder layer",
            )),
        }
    }

    pub(in crate::backend::model) fn execute_shared_prefill(
        &mut self,
        prefill: LayerPrefill<'_>,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        plan: &KvWritePlan,
        table: &BlockTable,
        start: usize,
    ) -> Result<()> {
        match (self, prefill) {
            (Self::Moe(layer), LayerPrefill::Moe(prefill)) => layer
                .decode
                .as_mut()
                .ok_or(Error::InvalidDecoderKernel("CUDA prefill executor belongs to model graph"))?
                .execute_prefill(prefill, input, plan, table, start, output),
            (Self::Dense(layer), LayerPrefill::Dense(prefill)) => {
                let weights = layer.template.weights();
                let decode = layer.decode.as_mut().ok_or(Error::InvalidDecoderKernel(
                    "CUDA prefill executor belongs to model graph",
                ))?;
                prefill.execute(decode, input, weights, plan, table, start, output)
            },
            _ => Err(Error::InvalidDecoderKernel(
                "shared CUDA prefill plan differs from decoder layer",
            )),
        }
    }
}
