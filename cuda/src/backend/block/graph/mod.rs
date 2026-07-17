use ::runtime::kv::{BlockTable, KvWritePlan};
use mircuda::{DeviceBuffer, Graph, Stream, bf16};

use super::{DecodeMoeBlockBf16, DecodeMoeBlockWeights};
use crate::{
    Error, Result,
    backend::attention::graph::{Configs, Dynamic, Geometry, Kernels, Nodes},
};

mod arguments;
mod execution;
mod layer;
pub(super) mod weights;

pub(in crate::backend) use layer::CapturedLayer;
use weights::CapturedBlockWeights;

/// Reusable CUDA Graph for one fixed-shape routed-MoE decoder layer.
pub struct CapturedDecodeMoeBlockBf16 {
    graph: Graph<Resources>,
    stream: Stream,
    nodes: Nodes,
    kernels: Kernels,
    configs: Configs,
}

struct Resources {
    layer: CapturedLayer,
    nodes: Option<Nodes>,
}

impl DecodeMoeBlockBf16 {
    /// Captures this prepared layer without launching it.
    pub fn capture(
        self,
        input: &DeviceBuffer<bf16>,
        weights: DecodeMoeBlockWeights<'_>,
        write_plan: &KvWritePlan,
        table: &BlockTable,
        output: &DeviceBuffer<bf16>,
    ) -> Result<CapturedDecodeMoeBlockBf16> {
        capture_block(self, input.clone(), weights.into(), write_plan, table, output.clone())
    }
}

impl CapturedDecodeMoeBlockBf16 {
    /// Replays the currently bound token.
    pub fn launch(&mut self) -> Result<()> {
        Ok(self.graph.launch(&self.stream)?)
    }

    /// Rebinds token-dependent attention state and enqueues one layer replay.
    pub fn execute(&mut self, write_plan: &KvWritePlan, table: &BlockTable) -> Result<()> {
        let dynamic = Dynamic::new(write_plan, table, self.graph.resources().layer.geometry())?;
        self.validate_blocks(table)?;
        self.graph.update_kernel(
            &self.nodes.qkv_postprocess,
            &self.kernels.qkv_postprocess,
            self.configs.qkv_postprocess,
            move |resources| {
                resources.layer.set_dynamic(dynamic);
                resources.layer.qkv_arguments()
            },
        )?;
        self.graph.update_kernel(
            &self.nodes.kv_store,
            &self.kernels.kv_store,
            self.configs.kv_store,
            |resources| resources.layer.kv_arguments(),
        )?;
        self.graph.update_kernel(
            &self.nodes.attention.direct,
            &self.kernels.attention.direct,
            self.configs.attention,
            |resources| resources.layer.attention_arguments(),
        )?;
        let split = self.graph.resources().layer.split_attention_configs()?;
        self.graph.try_update_kernel(
            &self.nodes.attention.split.split,
            &self.kernels.attention.split.split,
            split.split,
            |resources| resources.layer.split_attention_arguments(),
        )?;
        self.graph.try_update_kernel(
            &self.nodes.attention.split.merge,
            &self.kernels.attention.split.merge,
            split.merge,
            |resources| resources.layer.merge_attention_arguments(),
        )?;
        self.launch()
    }

    /// Whether the physical K/V mapping differs from this graph's capture.
    #[must_use]
    pub fn requires_recapture(&self, table: &BlockTable) -> bool {
        self.graph.resources().layer.requires_recapture(table)
    }

    /// Replaces the native graph while retaining weights, cache pages, and I/O
    /// allocations.
    pub fn recapture(self, write_plan: &KvWritePlan, table: &BlockTable) -> Result<Self> {
        Dynamic::new(write_plan, table, self.graph.resources().layer.geometry())?;
        let resources = self.graph.into_resources();
        tracing::debug!(
            layer = resources.layer.geometry().layer,
            token = table.token_len(),
            blocks = table.blocks().len(),
            "recapturing CUDA MoE block after KV mapping change"
        );
        capture_block(
            resources.layer.block,
            resources.layer.input,
            resources.layer.weights,
            write_plan,
            table,
            resources.layer.output,
        )
    }

    fn validate_blocks(&self, table: &BlockTable) -> Result<()> {
        if self.requires_recapture(table) {
            Err(Error::InvalidPagedKv("captured MoE block requires KV block-table recapture"))
        } else {
            Ok(())
        }
    }
}

pub(super) fn capture_block(
    block: DecodeMoeBlockBf16,
    input: DeviceBuffer<bf16>,
    weights: CapturedBlockWeights,
    write_plan: &KvWritePlan,
    table: &BlockTable,
    output: DeviceBuffer<bf16>,
) -> Result<CapturedDecodeMoeBlockBf16> {
    let kernels = Kernels::new(&block.attention);
    let configs = Configs::new(&block.attention)?;
    let stream = block.stream.clone();
    let resources = Resources {
        layer: CapturedLayer::new(block, input, weights, write_plan, table, output)?,
        nodes: None,
    };
    let graph = stream.capture(resources, capture)?;
    let nodes = graph
        .resources()
        .nodes
        .ok_or(Error::InvalidDecoderKernel("MoE block capture produced no dynamic nodes"))?;
    Ok(CapturedDecodeMoeBlockBf16 { graph, stream, nodes, kernels, configs })
}

fn capture(resources: &mut Resources) -> Result<()> {
    resources.nodes = Some(resources.layer.capture()?);
    Ok(())
}
