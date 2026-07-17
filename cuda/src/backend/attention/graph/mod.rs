use ::runtime::kv::{BlockTable, KvWritePlan};
use mircuda::{DeviceBuffer, Graph, KernelNode, LaunchConfig, Stream, TypedKernel, bf16};

use super::{DecodeAttentionBf16, DecodeAttentionWeights};
use crate::{
    Error, Result,
    backend::kv::{CapturedPagedAttentionKernels, CapturedPagedAttentionNodes},
    kernels::{KvStoreKernel, QkvPostprocessArguments, QkvPostprocessKernel},
};

mod arguments;
mod execution;
mod weights;

use arguments::{
    attention_arguments, kv_arguments, merge_attention_arguments, qkv_arguments,
    split_attention_arguments,
};
pub(in crate::backend) use weights::CapturedAttentionWeights;

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

/// Reusable one-token attention graph over fixed device allocations and
/// weights.
///
/// The physical block mapping is fixed for the lifetime of this graph. A caller
/// must capture a replacement when the sequence acquires or remaps a page.
pub struct CapturedDecodeAttentionBf16 {
    graph: Graph<Resources>,
    stream: Stream,
    nodes: Nodes,
    kernels: Kernels,
    configs: Configs,
}

struct Resources {
    attention: DecodeAttentionBf16,
    input: DeviceBuffer<bf16>,
    output: DeviceBuffer<bf16>,
    weights: CapturedAttentionWeights,
    write_plan: KvWritePlan,
    table: BlockTable,
    geometry: Geometry,
    dynamic: Dynamic,
    blocks: Vec<u32>,
    nodes: Option<Nodes>,
}

#[derive(Clone, Copy)]
pub(in crate::backend) struct Dynamic {
    pub(in crate::backend) position: u32,
    pub(in crate::backend) local_start: u32,
    pub(in crate::backend) physical_block: u32,
    pub(in crate::backend) page_start: u32,
    pub(in crate::backend) token_count: u32,
    pub(in crate::backend) block_count: u32,
}

#[derive(Clone, Copy)]
pub(in crate::backend) struct Geometry {
    pub(in crate::backend) layer: usize,
    pub(in crate::backend) block_size: usize,
    pub(in crate::backend) block_size_abi: u32,
    pub(in crate::backend) block_count: u32,
    pub(in crate::backend) query_heads: u32,
    pub(in crate::backend) kv_heads: u32,
    pub(in crate::backend) head_dim: u32,
    pub(in crate::backend) value_head_dim: u32,
    pub(in crate::backend) rotary_dim: u32,
    pub(in crate::backend) pairing_dim: u32,
    pub(in crate::backend) window: u32,
    pub(in crate::backend) scale: f32,
    pub(in crate::backend) split_threshold: u32,
    pub(in crate::backend) theta: f32,
    pub(in crate::backend) epsilon: f32,
    pub(in crate::backend) normalization: crate::kernels::QkvNormalization,
    pub(in crate::backend) separate_qkv: bool,
}

#[derive(Clone, Copy)]
pub(in crate::backend) struct Nodes {
    pub(in crate::backend) qkv_postprocess: KernelNode<QkvPostprocessKernel>,
    pub(in crate::backend) kv_store: KernelNode<KvStoreKernel>,
    pub(in crate::backend) attention: CapturedPagedAttentionNodes,
}

pub(in crate::backend) struct Kernels {
    pub(in crate::backend) qkv_postprocess: TypedKernel<QkvPostprocessKernel>,
    pub(in crate::backend) kv_store: TypedKernel<KvStoreKernel>,
    pub(in crate::backend) attention: CapturedPagedAttentionKernels,
}

pub(in crate::backend) struct Configs {
    pub(in crate::backend) qkv_postprocess: LaunchConfig,
    pub(in crate::backend) kv_store: LaunchConfig,
    pub(in crate::backend) attention: LaunchConfig,
}

impl DecodeAttentionBf16 {
    /// Captures this prepared attention layer without launching the graph.
    pub fn capture(
        mut self,
        input: &DeviceBuffer<bf16>,
        weights: DecodeAttentionWeights<'_>,
        write_plan: &KvWritePlan,
        table: &BlockTable,
        output: &DeviceBuffer<bf16>,
    ) -> Result<CapturedDecodeAttentionBf16> {
        let geometry = Geometry::new(&self)?;
        let dynamic = Dynamic::new(write_plan, table, geometry)?;
        self.attention.prepare_table(table)?;
        let kernels = Kernels::new(&self);
        let configs = Configs::new(&self)?;
        let stream = self.stream.clone();
        let resources = Resources {
            attention: self,
            input: input.clone(),
            output: output.clone(),
            weights: weights.into(),
            write_plan: write_plan.clone(),
            table: table.clone(),
            geometry,
            dynamic,
            blocks: table.blocks().iter().map(|block| block.0).collect(),
            nodes: None,
        };
        let graph = stream.capture(resources, capture)?;
        let nodes = graph
            .resources()
            .nodes
            .ok_or(Error::InvalidDecoderKernel("attention capture produced no dynamic nodes"))?;
        Ok(CapturedDecodeAttentionBf16 { graph, stream, nodes, kernels, configs })
    }
}

impl CapturedDecodeAttentionBf16 {
    /// Replays the currently bound token without changing graph arguments.
    pub fn launch(&mut self) -> Result<()> {
        Ok(self.graph.launch(&self.stream)?)
    }

    /// Rebinds token-dependent scalars and enqueues one graph replay.
    pub fn execute(&mut self, write_plan: &KvWritePlan, table: &BlockTable) -> Result<()> {
        let dynamic = Dynamic::new(write_plan, table, self.graph.resources().geometry)?;
        self.validate_blocks(table)?;
        self.graph.update_kernel(
            &self.nodes.qkv_postprocess,
            &self.kernels.qkv_postprocess,
            self.configs.qkv_postprocess,
            move |resources| {
                resources.dynamic = dynamic;
                qkv_arguments(resources)
            },
        )?;
        self.graph.update_kernel(
            &self.nodes.kv_store,
            &self.kernels.kv_store,
            self.configs.kv_store,
            kv_arguments,
        )?;
        self.graph.update_kernel(
            &self.nodes.attention.direct,
            &self.kernels.attention.direct,
            self.configs.attention,
            attention_arguments,
        )?;
        let split = self
            .graph
            .resources()
            .attention
            .attention
            .captured_configs(dynamic.token_count, self.graph.resources().geometry.window)?;
        self.graph.try_update_kernel(
            &self.nodes.attention.split.split,
            &self.kernels.attention.split.split,
            split.split,
            split_attention_arguments,
        )?;
        self.graph.try_update_kernel(
            &self.nodes.attention.split.merge,
            &self.kernels.attention.split.merge,
            split.merge,
            merge_attention_arguments,
        )?;
        self.launch()
    }

    fn validate_blocks(&self, table: &BlockTable) -> Result<()> {
        let blocks = &self.graph.resources().blocks;
        if table.blocks().len() != blocks.len()
            || table.blocks().iter().zip(blocks).any(|(block, expected)| block.0 != *expected)
        {
            Err(Error::InvalidPagedKv("captured attention requires KV block-table recapture"))
        } else {
            Ok(())
        }
    }
}

fn capture(resources: &mut Resources) -> Result<()> {
    resources.nodes = Some(resources.attention.execute_captured(
        &resources.input,
        resources.weights.borrow(),
        &resources.write_plan,
        &resources.table,
        &mut resources.output,
    )?);
    Ok(())
}
