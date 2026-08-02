use mircuda::{DeviceBuffer, bf16};
use runtime::kv::{BlockTable, KvWritePlan};

use super::super::layer::{LayerPrefill, PreparedLayer};
use crate::{
    PagedPrefillBatch, Result,
    backend::{
        attention::graph::{Configs, Dynamic, Geometry, Kernels, Nodes},
        block::CapturedLayer,
        dense::graph::CapturedDenseLayer,
    },
    kernels::{
        MergeAttentionArguments, QkvPostprocessArguments, SplitAttentionArguments,
        SplitAttentionConfigs,
    },
};

pub(super) enum CapturedModelLayer {
    Moe(Box<CapturedLayer>),
    Dense(Box<CapturedDenseLayer>),
}

impl CapturedModelLayer {
    pub fn new(prepared: PreparedLayer, plan: &KvWritePlan, table: &BlockTable) -> Result<Self> {
        match prepared {
            PreparedLayer::Moe(prepared) => {
                let prepared = *prepared;
                Ok(Self::Moe(Box::new(CapturedLayer::new(
                    prepared.block, prepared.input, prepared.weights, plan, table, prepared.output,
                )?)))
            },
            PreparedLayer::Dense(prepared) => {
                Ok(Self::Dense(Box::new(CapturedDenseLayer::new(*prepared, plan, table)?)))
            },
        }
    }

    pub fn capture(&mut self) -> Result<Nodes> {
        match self {
            Self::Moe(layer) => layer.capture(),
            Self::Dense(layer) => layer.capture(),
        }
    }

    pub fn prepare_replay(&mut self, plan: &KvWritePlan, table: &BlockTable) -> Result<()> {
        match self {
            Self::Moe(layer) => layer.prepare_replay(plan, table),
            Self::Dense(layer) => layer.prepare_replay(plan, table),
        }
    }

    pub fn kernels(&self) -> Kernels {
        match self {
            Self::Moe(layer) => layer.kernels(),
            Self::Dense(layer) => layer.kernels(),
        }
    }

    pub fn configs(&self) -> Result<Configs> {
        match self {
            Self::Moe(layer) => layer.configs(),
            Self::Dense(layer) => layer.configs(),
        }
    }

    pub const fn geometry(&self) -> Geometry {
        match self {
            Self::Moe(layer) => layer.geometry(),
            Self::Dense(layer) => layer.geometry(),
        }
    }

    pub const fn set_dynamic(&mut self, dynamic: Dynamic) {
        match self {
            Self::Moe(layer) => layer.set_dynamic(dynamic),
            Self::Dense(layer) => layer.set_dynamic(dynamic),
        }
    }

    pub fn qkv_arguments(&mut self) -> QkvPostprocessArguments<'_> {
        match self {
            Self::Moe(layer) => layer.qkv_arguments(),
            Self::Dense(layer) => layer.qkv_arguments(),
        }
    }

    pub fn kv_arguments(&mut self) -> KvArguments<'_> {
        match self {
            Self::Moe(layer) => layer.kv_arguments(),
            Self::Dense(layer) => layer.kv_arguments(),
        }
    }

    pub fn attention_arguments(&mut self) -> AttentionArguments<'_> {
        match self {
            Self::Moe(layer) => layer.attention_arguments(),
            Self::Dense(layer) => layer.attention_arguments(),
        }
    }

    pub fn split_attention_configs(&self) -> Result<SplitAttentionConfigs> {
        match self {
            Self::Moe(layer) => layer.split_attention_configs(),
            Self::Dense(layer) => layer.split_attention_configs(),
        }
    }

    pub fn split_attention_arguments(&mut self) -> Result<SplitAttentionArguments<'_>> {
        match self {
            Self::Moe(layer) => layer.split_attention_arguments(),
            Self::Dense(layer) => layer.split_attention_arguments(),
        }
    }

    pub fn merge_attention_arguments(&mut self) -> Result<MergeAttentionArguments<'_>> {
        match self {
            Self::Moe(layer) => layer.merge_attention_arguments(),
            Self::Dense(layer) => layer.merge_attention_arguments(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute_prefill_masked(
        &mut self,
        prefill: LayerPrefill<'_>,
        input: &DeviceBuffer<bf16>,
        plan: &KvWritePlan,
        table: &BlockTable,
        start_position: usize,
        output: &mut DeviceBuffer<bf16>,
        image: Option<crate::backend::attention::ImageAttentionSpan>,
    ) -> Result<()> {
        match (self, prefill) {
            (Self::Moe(layer), LayerPrefill::Moe(prefill)) => layer
                .execute_prefill_masked(prefill, input, plan, table, start_position, output, image),
            (Self::Dense(layer), LayerPrefill::Dense(prefill)) => {
                if image.is_some() {
                    return Err(crate::Error::UnsupportedVisionContract(
                        "bidirectional pooled-image prefill requires a CUDA hybrid-MoE decoder"
                            .into(),
                    ));
                }
                layer.execute_prefill(prefill, input, plan, table, start_position, output)
            },
            _ => Err(crate::Error::InvalidDecoderKernel("prefill layer kind differs from graph")),
        }
    }

    pub fn execute_prefill_batch(
        &mut self,
        prefill: LayerPrefill<'_>,
        input: &DeviceBuffer<bf16>,
        batch: &PagedPrefillBatch,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        match (self, prefill) {
            (Self::Moe(layer), LayerPrefill::Moe(prefill)) => {
                layer.execute_prefill_batch(prefill, input, batch, output)
            },
            (Self::Dense(layer), LayerPrefill::Dense(prefill)) => {
                layer.execute_prefill_batch(prefill, input, batch, output)
            },
            _ => Err(crate::Error::InvalidDecoderKernel("prefill layer kind differs from graph")),
        }
    }
}

type KvArguments<'a> = (
    &'a DeviceBuffer<bf16>,
    &'a DeviceBuffer<bf16>,
    &'a mut DeviceBuffer<u8>,
    &'a mut DeviceBuffer<u8>,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
);

type AttentionArguments<'a> = (
    &'a DeviceBuffer<bf16>,
    &'a DeviceBuffer<u8>,
    &'a DeviceBuffer<u8>,
    &'a DeviceBuffer<u32>,
    &'a mut DeviceBuffer<bf16>,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    f32,
    u32,
);
