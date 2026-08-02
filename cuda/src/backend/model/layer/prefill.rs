use mircuda::{DeviceBuffer, bf16};
use runtime::kv::{BlockTable, KvWritePlan};

use super::{LayerPrefill, SessionLayer};
use crate::{Error, Result};

impl SessionLayer {
    pub(in crate::backend::model) fn prepare_prefill(&mut self, tokens: usize) -> Result<()> {
        match self {
            Self::Moe(layer) => {
                if !layer.prefill.contains_key(&tokens) {
                    layer.prefill.insert(tokens, layer.template.instantiate_prefill(tokens)?);
                }
            },
            Self::Dense(layer) => {
                if !layer.prefill.contains_key(&tokens) {
                    layer.prefill.insert(tokens, layer.template.instantiate_prefill(tokens)?);
                }
            },
        }
        Ok(())
    }

    pub(in crate::backend::model) fn prefill_plan(
        &mut self,
        tokens: usize,
    ) -> Result<LayerPrefill<'_>> {
        match self {
            Self::Moe(layer) => Ok(LayerPrefill::Moe(
                layer
                    .prefill
                    .get_mut(&tokens)
                    .ok_or(Error::InvalidDecoderKernel("missing CUDA prefill plan"))?,
            )),
            Self::Dense(layer) => Ok(LayerPrefill::Dense(
                layer
                    .prefill
                    .get_mut(&tokens)
                    .ok_or(Error::InvalidDecoderKernel("missing dense CUDA prefill plan"))?,
            )),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::backend::model) fn execute_prefill_masked(
        &mut self,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        write_plan: &KvWritePlan,
        table: &BlockTable,
        start_position: usize,
        tokens: usize,
        image: Option<crate::backend::attention::ImageAttentionSpan>,
    ) -> Result<()> {
        match self {
            Self::Moe(layer) => {
                let decode = layer.decode.as_mut().ok_or(Error::InvalidDecoderKernel(
                    "CUDA prefill executor belongs to model graph",
                ))?;
                let prefill = layer
                    .prefill
                    .get_mut(&tokens)
                    .ok_or(Error::InvalidDecoderKernel("missing CUDA prefill plan"))?;
                decode.execute_prefill_masked(
                    prefill, input, write_plan, table, start_position, output, image,
                )
            },
            Self::Dense(layer) => {
                if image.is_some() {
                    return Err(Error::UnsupportedVisionContract(
                        "bidirectional pooled-image prefill requires a CUDA hybrid-MoE decoder"
                            .into(),
                    ));
                }
                let weights = layer.template.weights();
                let decode = layer.decode.as_mut().ok_or(Error::InvalidDecoderKernel(
                    "CUDA prefill executor belongs to model graph",
                ))?;
                let prefill = layer
                    .prefill
                    .get_mut(&tokens)
                    .ok_or(Error::InvalidDecoderKernel("missing dense CUDA prefill plan"))?;
                prefill.execute(decode, input, weights, write_plan, table, start_position, output)
            },
        }
    }
}
