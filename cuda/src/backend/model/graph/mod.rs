use mircuda::{DeviceBuffer, Graph, Stream, bf16};
use runtime::kv::{BlockTable, KvWritePlan};
use uuid::Uuid;

use super::layer::{LayerPrefill, PreparedLayer, SessionLayer};
use crate::{
    Error, Result,
    backend::attention::graph::{Configs, Dynamic, Kernels, Nodes},
};

mod layer;

use layer::CapturedModelLayer;

struct Resources {
    layers: Vec<CapturedModelLayer>,
    nodes: Option<Vec<Nodes>>,
}

struct Binding {
    nodes: Nodes,
    kernels: Kernels,
    configs: Configs,
}

/// One CUDA Graph replay containing every decoder layer for one session.
pub(super) struct CapturedModelDecode {
    graph: Graph<Resources>,
    stream: Stream,
    bindings: Vec<Binding>,
}

impl CapturedModelDecode {
    pub(super) fn new(
        stream: Stream,
        prepared: Vec<PreparedLayer>,
        session_id: Uuid,
        table: &BlockTable,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(prepared.len());
        for prepared in prepared {
            let plan = KvWritePlan::prefill(
                session_id,
                prepared.layer(),
                table,
                table.token_len() - 1,
                1,
            )?;
            layers.push(CapturedModelLayer::new(prepared, &plan, table)?);
        }
        capture(stream, layers)
    }

    pub(super) fn execute(&mut self, session_id: Uuid, table: &BlockTable) -> Result<()> {
        for index in 0..self.bindings.len() {
            let geometry = self.graph.resources().layers[index].geometry();
            let plan =
                KvWritePlan::prefill(session_id, geometry.layer, table, table.token_len() - 1, 1)?;
            let dynamic = Dynamic::new(&plan, table, geometry)?;
            self.graph.with_resources_mut(|resources| {
                resources.layers[index].prepare_replay(&plan, table)
            })?;
            let binding = &self.bindings[index];
            self.graph.update_kernel(
                &binding.nodes.qkv_postprocess,
                &binding.kernels.qkv_postprocess,
                binding.configs.qkv_postprocess,
                move |resources| {
                    resources.layers[index].set_dynamic(dynamic);
                    resources.layers[index].qkv_arguments()
                },
            )?;
            self.graph.update_kernel(
                &binding.nodes.kv_store,
                &binding.kernels.kv_store,
                binding.configs.kv_store,
                move |resources| resources.layers[index].kv_arguments(),
            )?;
            self.graph.update_kernel(
                &binding.nodes.attention.direct,
                &binding.kernels.attention.direct,
                binding.configs.attention,
                move |resources| resources.layers[index].attention_arguments(),
            )?;
            let split = self.graph.resources().layers[index].split_attention_configs()?;
            self.graph.try_update_kernel(
                &binding.nodes.attention.split.split,
                &binding.kernels.attention.split.split,
                split.split,
                move |resources| resources.layers[index].split_attention_arguments(),
            )?;
            self.graph.try_update_kernel(
                &binding.nodes.attention.split.merge,
                &binding.kernels.attention.split.merge,
                split.merge,
                move |resources| resources.layers[index].merge_attention_arguments(),
            )?;
        }
        Ok(self.graph.launch(&self.stream)?)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn execute_prefill(
        &mut self,
        index: usize,
        prefill: LayerPrefill<'_>,
        input: &DeviceBuffer<bf16>,
        write_plan: &KvWritePlan,
        table: &BlockTable,
        start_position: usize,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.graph.with_resources_mut(|resources| {
            resources.layers[index]
                .execute_prefill(prefill, input, write_plan, table, start_position, output)
        })
    }
}

pub(super) fn execute_layers(
    graph: &mut Option<CapturedModelDecode>,
    layers: &mut [SessionLayer],
    stream: &Stream,
    session_id: Uuid,
    table: &BlockTable,
) -> Result<()> {
    let position = table
        .token_len()
        .checked_sub(1)
        .ok_or(Error::InvalidPagedKv("decode requires a non-empty block table"))?;
    if let Some(captured) = graph {
        captured.execute(session_id, table)?;
        return Ok(());
    }
    for layer in &mut *layers {
        let plan = KvWritePlan::prefill(session_id, layer.layer(), table, position, 1)?;
        layer.decode_direct(&plan, table)?;
    }
    let prepared =
        layers.iter_mut().map(SessionLayer::take_prepared).collect::<Result<Vec<_>>>()?;
    *graph = Some(CapturedModelDecode::new(stream.clone(), prepared, session_id, table)?);
    Ok(())
}

fn capture(stream: Stream, layers: Vec<CapturedModelLayer>) -> Result<CapturedModelDecode> {
    if layers.is_empty() {
        return Err(Error::InvalidDecoderKernel("model graph requires decoder layers"));
    }
    let resources = Resources { layers, nodes: None };
    let graph = stream.capture(resources, capture_layers)?;
    let nodes = graph
        .resources()
        .nodes
        .as_ref()
        .ok_or(Error::InvalidDecoderKernel("model capture produced no dynamic nodes"))?;
    let bindings = graph
        .resources()
        .layers
        .iter()
        .zip(nodes)
        .map(|(layer, nodes)| {
            Ok(Binding {
                nodes: *nodes,
                kernels: layer.kernels(),
                configs: layer.configs()?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    tracing::debug!(layers = bindings.len(), "captured full CUDA model decode graph");
    Ok(CapturedModelDecode { graph, stream, bindings })
}

fn capture_layers(resources: &mut Resources) -> Result<()> {
    resources.nodes = Some(
        resources
            .layers
            .iter_mut()
            .map(CapturedModelLayer::capture)
            .collect::<Result<Vec<_>>>()?,
    );
    Ok(())
}
