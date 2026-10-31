use mircuda::{DeviceBuffer, bf16};

use super::{
    super::SharedRoutedLayerTemplate,
    states::{attention_states, full_states, linear_states, mamba_states},
};
use crate::{
    CudaAffineGatedDeltaMoeExecution, CudaAffineGatedFullAttentionMoeExecution,
    CudaAttentionHybridExecution, CudaMamba2HybridExecution, CudaSharedRoutedModelSession,
    ExecutionPhase, PagedDecodeBatch, Result, kernels::BatchedSplitAttentionWorkspace,
};

#[derive(Debug)]
pub(super) enum SharedRoutedBatchLayer {
    Linear(Box<CudaAffineGatedDeltaMoeExecution>),
    Full(Box<CudaAffineGatedFullAttentionMoeExecution>),
    Mamba2(Box<CudaMamba2HybridExecution>),
    Attention(Box<CudaAttentionHybridExecution>),
}

impl SharedRoutedBatchLayer {
    /// `capacity` is the row count the recurrent layers' shared packed states
    /// are allocated for, so every batch up to it reuses one allocation.
    pub(super) fn new(
        template: &SharedRoutedLayerTemplate,
        rows: usize,
        capacity: usize,
        phase: ExecutionPhase,
        workspace: Option<BatchedSplitAttentionWorkspace>,
    ) -> Result<Self> {
        Ok(match template {
            SharedRoutedLayerTemplate::Linear(layer) => {
                let mut execution = layer.prepare_phase(rows, phase)?;
                execution.set_batch_capacity(capacity);
                Self::Linear(Box::new(execution))
            },
            SharedRoutedLayerTemplate::Full(layer) => {
                Self::Full(Box::new(layer.prepare_phase_with_workspace(rows, phase, workspace)?))
            },
            SharedRoutedLayerTemplate::Mamba2(layer) => {
                let mut execution = layer.prepare_phase(rows, phase)?;
                execution.set_batch_capacity(capacity);
                Self::Mamba2(Box::new(execution))
            },
            SharedRoutedLayerTemplate::Attention(layer) => {
                let mut execution = layer.prepare_phase(rows, phase)?;
                execution.set_workspace(workspace);
                Self::Attention(Box::new(execution))
            },
        })
    }

    /// Follows a K/V resize to `block_count` blocks.
    pub(super) fn follow_cache_blocks(&mut self, block_count: u32) {
        if let Self::Full(execution) = self {
            execution.follow_cache_blocks(block_count);
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
                execution.prepare_packed(input, &linear_states(sessions, layer)?, output)
            },
            Self::Full(execution) => {
                execution.prepare_packed(input, &full_states(sessions, layer)?, paging, output)
            },
            Self::Mamba2(execution) => execution.prepare_packed(&mamba_states(sessions, layer)?),
            Self::Attention(execution) => {
                execution.prepare_packed(&attention_states(sessions, layer)?, paging)
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
            Self::Mamba2(execution) => execution.execute_prepared_packed(input, output),
            Self::Attention(execution) => execution.execute_prepared_packed(input, paging, output),
        }
    }

    pub(super) fn commit(
        &mut self,
        sessions: &mut [&mut CudaSharedRoutedModelSession],
        layer: usize,
    ) -> Result<()> {
        match self {
            Self::Linear(execution) => {
                execution.commit_packed(&mut linear_states(sessions, layer)?)
            },
            Self::Mamba2(execution) => execution.commit_packed(&mut mamba_states(sessions, layer)?),
            Self::Full(_) | Self::Attention(_) => Ok(()),
        }
    }

    pub(super) fn capture_partitions(&self, paging: &PagedDecodeBatch) -> usize {
        match self {
            Self::Linear(_) | Self::Mamba2(_) => 0,
            Self::Full(execution) => execution.packed_capture_partitions(paging),
            Self::Attention(execution) => execution.packed_capture_partitions(paging),
        }
    }
}
