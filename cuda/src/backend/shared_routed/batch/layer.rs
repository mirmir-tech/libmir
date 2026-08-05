use mircuda::{DeviceBuffer, bf16};

use super::super::SharedRoutedLayerTemplate;
use crate::{
    CudaAffineGatedDeltaMoeExecution, CudaAffineGatedFullAttentionMoeExecution,
    CudaSharedRoutedModelSession, ExecutionPhase, PagedDecodeBatch, Result,
    kernels::BatchedSplitAttentionWorkspace,
};

#[derive(Debug)]
pub(super) enum SharedRoutedBatchLayer {
    Linear(Box<CudaAffineGatedDeltaMoeExecution>),
    Full(Box<CudaAffineGatedFullAttentionMoeExecution>),
}

impl SharedRoutedBatchLayer {
    pub(super) fn new(
        template: &SharedRoutedLayerTemplate,
        rows: usize,
        phase: ExecutionPhase,
        workspace: Option<BatchedSplitAttentionWorkspace>,
    ) -> Result<Self> {
        match template {
            SharedRoutedLayerTemplate::Linear(layer) => {
                layer.prepare_phase(rows, phase).map(Box::new).map(Self::Linear)
            },
            SharedRoutedLayerTemplate::Full(layer) => layer
                .prepare_phase_with_workspace(rows, phase, workspace)
                .map(Box::new)
                .map(Self::Full),
        }
    }

    pub(super) fn prepare(
        &mut self,
        input: &DeviceBuffer<bf16>,
        output: &DeviceBuffer<bf16>,
        sessions: &mut [&mut CudaSharedRoutedModelSession],
        layer: usize,
        paging: &PagedDecodeBatch,
    ) -> Result<()> {
        match self {
            Self::Linear(execution) => {
                let states = super::states::linear_states(sessions, layer)?;
                execution.prepare_packed(input, &states, output)
            },
            Self::Full(execution) => {
                let states = super::states::full_states(sessions, layer)?;
                execution.prepare_packed(input, &states, paging, output)
            },
        }
    }

    pub(super) fn execute_prepared(
        &mut self,
        input: &DeviceBuffer<bf16>,
        positions: &DeviceBuffer<u32>,
        paging: &PagedDecodeBatch,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        match self {
            Self::Linear(execution) => execution.execute_prepared_packed(input, output),
            Self::Full(execution) => {
                execution.execute_prepared_packed(input, positions, paging, output)
            },
        }
    }

    pub(super) fn commit(
        &mut self,
        sessions: &mut [&mut CudaSharedRoutedModelSession],
        layer: usize,
    ) -> Result<()> {
        if let Self::Linear(execution) = self {
            let mut states = super::states::linear_states(sessions, layer)?;
            execution.commit_packed(&mut states)?;
        }
        Ok(())
    }

    pub(super) fn capture_partitions(&self, paging: &PagedDecodeBatch) -> usize {
        match self {
            Self::Linear(_) => 0,
            Self::Full(execution) => execution.packed_capture_partitions(paging),
        }
    }
}
