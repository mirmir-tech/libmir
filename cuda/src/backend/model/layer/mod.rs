use std::collections::HashMap;

mod prefill;
mod prepared;
mod shared;

use mircuda::{DeviceBuffer, bf16};
pub(super) use prepared::{LayerPrefill, PrefillSignature, PreparedLayer, SharedLayerPrefill};
use runtime::kv::{BlockTable, KvWritePlan};

use crate::{
    BatchedDecodeMoeLayer, BatchedPagedAttentionBf16, CudaBackend, DecodeDenseSwiGlu,
    DecodeMoeBlockExecutor, DecodeMoeLayerTemplate, DenseSwiGluLayerTemplate, Error,
    PagedDecodeBatch, PagedKvCache, PrefillDenseSwiGlu, PrefillMoeBlockBf16, Result,
    backend::dense::{
        BatchedDecodeDenseLayer,
        graph::{DenseSwiGluWeightsOwned, PreparedDecodeDense},
    },
    kernels::BatchedSplitAttentionWorkspace,
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
        workspace: BatchedSplitAttentionWorkspace,
    ) -> Result<BatchedLayer> {
        match self {
            Self::Moe(template) => Ok(BatchedLayer::Moe(Box::new(
                template.instantiate_batch_with_cache_workspace(rows, cache, Some(workspace))?,
            ))),
            Self::Dense(template) => Ok(BatchedLayer::Dense(Box::new(
                template.instantiate_batch_with_cache_workspace(rows, cache, Some(workspace))?,
            ))),
        }
    }
}

pub(super) fn allocate_batch_attention_workspace(
    backend: &CudaBackend,
    layers: &[DecoderLayerTemplate],
    caches: &[PagedKvCache],
    rows: usize,
) -> Result<BatchedSplitAttentionWorkspace> {
    let (values, statistics) = layers.iter().zip(caches).try_fold(
        (0_usize, 0_usize),
        |(values, statistics), (layer, cache)| {
            let attention = layer.attention();
            let required = BatchedPagedAttentionBf16::workspace_lengths(
                backend,
                cache,
                attention.query_heads,
                attention.max_sequence_blocks,
                rows,
            )?;
            Ok::<_, Error>((values.max(required.0), statistics.max(required.1)))
        },
    )?;
    if values == 0 || statistics == 0 {
        return Err(Error::InvalidDecoderKernel("CUDA batch requires at least one layer"));
    }
    Ok(BatchedSplitAttentionWorkspace::new(
        backend.inner.pool.allocate(&backend.inner.stream, values)?,
        backend.inner.pool.allocate(&backend.inner.stream, statistics)?,
        backend.inner.pool.allocate(&backend.inner.stream, statistics)?,
    ))
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

    pub(super) const fn layer(&self) -> usize {
        match self {
            Self::Moe(layer) => layer.template.config().attention.layer,
            Self::Dense(layer) => layer.template.config().attention.layer,
        }
    }

    pub(super) const fn attention_config(&self) -> crate::DecodeAttentionConfig {
        match self {
            Self::Moe(layer) => layer.template.config().attention,
            Self::Dense(layer) => layer.template.config().attention,
        }
    }
}
