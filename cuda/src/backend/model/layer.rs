use std::collections::HashMap;

use mircuda::{DeviceBuffer, bf16};
use runtime::kv::{BlockTable, KvWritePlan};

use crate::{
    BatchedDecodeMoeLayer, DecodeDenseSwiGlu, DecodeMoeBlockExecutor, DecodeMoeLayerTemplate,
    DenseSwiGluLayerTemplate, Error, PagedDecodeBatch, PagedKvCache, PrefillDenseSwiGlu,
    PrefillMoeBlockBf16, Result,
    backend::{
        block::PreparedDecodeMoeBlock,
        dense::{
            BatchedDecodeDenseLayer,
            graph::{DenseSwiGluWeightsOwned, PreparedDecodeDense},
        },
    },
};

#[derive(Clone)]
pub(super) enum DecoderLayerTemplate {
    Moe(Box<DecodeMoeLayerTemplate>),
    Dense(Box<DenseSwiGluLayerTemplate>),
}

pub(super) enum SessionLayer {
    Moe(Box<MoeLayer>),
    Dense(Box<DenseLayer>),
}

pub(super) enum BatchedLayer {
    Moe(Box<BatchedDecodeMoeLayer>),
    Dense(Box<BatchedDecodeDenseLayer>),
}

pub(super) struct MoeLayer {
    template: DecodeMoeLayerTemplate,
    decode: Option<DecodeMoeBlockExecutor>,
    prefill: HashMap<usize, PrefillMoeBlockBf16>,
}

pub(super) struct DenseLayer {
    template: DenseSwiGluLayerTemplate,
    decode: Option<DecodeDenseSwiGlu>,
    prefill: HashMap<usize, PrefillDenseSwiGlu>,
    input: DeviceBuffer<bf16>,
    output: DeviceBuffer<bf16>,
}

impl DecoderLayerTemplate {
    pub(super) fn instantiate(
        &self,
        input: &DeviceBuffer<bf16>,
        output: &DeviceBuffer<bf16>,
        cache: PagedKvCache,
    ) -> Result<SessionLayer> {
        match self {
            Self::Moe(template) => Ok(SessionLayer::Moe(Box::new(MoeLayer {
                decode: Some(template.instantiate_with_cache(input, output, cache)?),
                template: template.as_ref().clone(),
                prefill: HashMap::new(),
            }))),
            Self::Dense(template) => Ok(SessionLayer::Dense(Box::new(DenseLayer {
                decode: Some(template.instantiate_with_cache(input, output, cache)?),
                template: template.as_ref().clone(),
                prefill: HashMap::new(),
                input: input.clone(),
                output: output.clone(),
            }))),
        }
    }

    pub(super) const fn attention(&self) -> crate::DecodeAttentionConfig {
        match self {
            Self::Moe(template) => template.config().attention,
            Self::Dense(template) => template.config().attention,
        }
    }

    pub(super) fn instantiate_batch(
        &self,
        rows: usize,
        cache: PagedKvCache,
    ) -> Result<BatchedLayer> {
        match self {
            Self::Moe(template) => {
                Ok(BatchedLayer::Moe(Box::new(template.instantiate_batch_with_cache(rows, cache)?)))
            },
            Self::Dense(template) => Ok(BatchedLayer::Dense(Box::new(
                template.instantiate_batch_with_cache(rows, cache)?,
            ))),
        }
    }
}

impl BatchedLayer {
    pub(super) fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        batch: &PagedDecodeBatch,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        match self {
            Self::Moe(layer) => layer.execute(input, batch, output),
            Self::Dense(layer) => layer.execute(input, batch, output),
        }
    }
}

impl SessionLayer {
    pub(super) fn decode_direct(
        &mut self,
        write_plan: &KvWritePlan,
        table: &BlockTable,
    ) -> Result<()> {
        match self {
            Self::Moe(layer) => layer
                .decode
                .as_mut()
                .ok_or(Error::InvalidDecoderKernel("model layer already belongs to a graph"))?
                .execute_direct(write_plan, table),
            Self::Dense(layer) => {
                let weights = layer.template.weights();
                layer
                    .decode
                    .as_mut()
                    .ok_or(Error::InvalidDecoderKernel("model layer already belongs to a graph"))?
                    .execute(&layer.input, weights, write_plan, table, &mut layer.output)
            },
        }
    }

    pub(super) fn take_prepared(&mut self) -> Result<PreparedLayer> {
        match self {
            Self::Moe(layer) => Ok(PreparedLayer::Moe(Box::new(
                layer
                    .decode
                    .take()
                    .ok_or(Error::InvalidDecoderKernel("model layer already belongs to a graph"))?
                    .into_prepared()?,
            ))),
            Self::Dense(layer) => Ok(PreparedLayer::Dense(Box::new(PreparedDecodeDense {
                block: layer
                    .decode
                    .take()
                    .ok_or(Error::InvalidDecoderKernel("model layer already belongs to a graph"))?,
                input: layer.input.clone(),
                output: layer.output.clone(),
                weights: DenseSwiGluWeightsOwned::try_from(layer.template.weights())?,
            }))),
        }
    }

    pub(super) fn prepare_prefill(&mut self, tokens: usize) -> Result<()> {
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

    pub(super) fn prefill_plan(&mut self, tokens: usize) -> Result<LayerPrefill<'_>> {
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
    pub(super) fn execute_prefill(
        &mut self,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        write_plan: &KvWritePlan,
        table: &BlockTable,
        start_position: usize,
        tokens: usize,
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
                decode.execute_prefill(prefill, input, write_plan, table, start_position, output)
            },
            Self::Dense(layer) => {
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

    pub(super) const fn layer(&self) -> usize {
        match self {
            Self::Moe(layer) => layer.template.config().attention.layer,
            Self::Dense(layer) => layer.template.config().attention.layer,
        }
    }
}

pub(super) enum PreparedLayer {
    Moe(Box<PreparedDecodeMoeBlock>),
    Dense(Box<PreparedDecodeDense>),
}

impl PreparedLayer {
    pub const fn layer(&self) -> usize {
        match self {
            Self::Moe(prepared) => prepared.layer,
            Self::Dense(prepared) => prepared.block.attention.config.layer,
        }
    }
}

pub(super) enum LayerPrefill<'a> {
    Moe(&'a mut PrefillMoeBlockBf16),
    Dense(&'a mut PrefillDenseSwiGlu),
}
