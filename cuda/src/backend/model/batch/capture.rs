use mircuda::{DeviceBuffer, Graph, bf16};
use runtime::kv::BlockTable;

use super::{BatchResources, CudaDecodeBatch, execute};
use crate::{Error, Result};

pub(super) enum BatchState {
    Direct(BatchResources),
    /// `partitions` holds what each layer's captured attention launch depends
    /// on; when a longer context changes any of them, the graph is captured
    /// again. Layers are compared one by one: a `FlashAttention` context band
    /// in one layer must not hide a partition change in another.
    Captured {
        graph: Graph<BatchResources>,
        partitions: Vec<usize>,
    },
}

impl BatchResources {
    fn capture_partitions(&self) -> Vec<usize> {
        self.layers.iter().map(|layer| layer.capture_partitions(&self.paging)).collect()
    }
}

impl CudaDecodeBatch {
    pub fn decode(
        &mut self,
        tokens: &[u32],
        tables: &[&BlockTable],
    ) -> Result<&DeviceBuffer<bf16>> {
        self.prepare(tokens, tables)?;
        let state = self
            .state
            .take()
            .ok_or(Error::InvalidDecoderKernel("CUDA batch state is unavailable"))?;
        let resources = match state {
            BatchState::Captured { mut graph, partitions }
                if graph.resources().capture_partitions() == partitions =>
            {
                graph.launch(&self.stream)?;
                self.state = Some(BatchState::Captured { graph, partitions });
                return self.logits_result();
            },
            BatchState::Captured { graph, .. } => graph.into_resources(),
            BatchState::Direct(resources) => resources,
        };
        let mut resources = resources;
        execute(&mut resources)?;
        let partitions = resources.capture_partitions();
        let graph = self.stream.capture(resources, execute)?;
        tracing::debug!(rows = self.batch_size, ?partitions, "captured CUDA model decode batch");
        self.state = Some(BatchState::Captured { graph, partitions });
        self.logits_result()
    }
}
