use mircuda::{DeviceBuffer, bf16};
use runtime::kv::{BlockTable, KvWritePlan};

use super::{LayerPrefill, PrefillSignature, SessionLayer, SharedLayerPrefill};
use crate::{Error, PagedPrefillBatch, Result, backend::dense::DenseLayerLayout};

impl SessionLayer {
    pub(in crate::backend::model) fn prefill_signature(&self) -> PrefillSignature {
        match self {
            Self::Moe(layer) => {
                let mut config = layer.template.config();
                if layer.template.experts_bind_at_execution() {
                    config.attention.layer = 0;
                }
                let (attention, experts) = layer.template.layouts();
                PrefillSignature::Moe(config, attention, experts)
            },
            Self::Dense(layer) => {
                let mut config = layer.template.config();
                if !config.binds_layer_weights() {
                    config.attention.layer = 0;
                }
                let layout = DenseLayerLayout::of(layer.template.weights());
                PrefillSignature::Dense(config, layout)
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
        Ok(SharedLayerPrefill::new(&signature, plan))
    }

    pub(in crate::backend::model) fn execute_shared_prefill_batch(
        &mut self,
        prefill: LayerPrefill<'_>,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        batch: &PagedPrefillBatch,
        per_layer: Option<&DeviceBuffer<bf16>>,
    ) -> Result<()> {
        match (self, prefill) {
            (Self::Moe(_), _) if per_layer.is_some() => Err(moe_per_layer()),
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
                prefill.execute_batch(decode, input, weights, batch, per_layer, output)
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
        (start, per_layer): (usize, Option<&DeviceBuffer<bf16>>),
    ) -> Result<()> {
        match (self, prefill) {
            (Self::Moe(_), _) if per_layer.is_some() => Err(moe_per_layer()),
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
                prefill.execute(decode, input, weights, plan, table, (start, per_layer), output)
            },
            _ => Err(Error::InvalidDecoderKernel(
                "shared CUDA prefill plan differs from decoder layer",
            )),
        }
    }
}

fn moe_per_layer() -> Error {
    Error::InvalidDecoderKernel("CUDA MoE layers take no per-layer input")
}
