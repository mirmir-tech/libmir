use mircuda::{DeviceBuffer, bf16};
use runtime::backend::DecodeSequence;

use super::super::SharedRoutedLayerTemplate;
use crate::{
    CudaAffineGatedDeltaMoeExecution, CudaAffineGatedFullAttentionMoeExecution,
    CudaSharedRoutedModelSession, Result,
};

#[derive(Debug)]
pub(super) enum SharedRoutedBatchLayer {
    Linear(Box<CudaAffineGatedDeltaMoeExecution>),
    Full(Box<CudaAffineGatedFullAttentionMoeExecution>),
}

impl SharedRoutedBatchLayer {
    pub(super) fn new(template: &SharedRoutedLayerTemplate, rows: usize) -> Result<Self> {
        match template {
            SharedRoutedLayerTemplate::Linear(layer) => {
                layer.prepare(rows).map(Box::new).map(Self::Linear)
            },
            SharedRoutedLayerTemplate::Full(layer) => {
                layer.prepare(rows).map(Box::new).map(Self::Full)
            },
        }
    }

    pub(super) fn prepare(
        &mut self,
        input: &DeviceBuffer<bf16>,
        output: &DeviceBuffer<bf16>,
        sessions: &mut [&mut CudaSharedRoutedModelSession],
        sequences: &[DecodeSequence],
        layer: usize,
        max_blocks: usize,
    ) -> Result<()> {
        match self {
            Self::Linear(execution) => {
                let states = super::states::linear_states(sessions, layer)?;
                execution.prepare_packed(input, &states, output)
            },
            Self::Full(execution) => {
                let states = super::states::full_states(sessions, layer)?;
                let tables =
                    sequences.iter().map(|sequence| &sequence.block_table).collect::<Vec<_>>();
                execution.prepare_packed(input, &states, &tables, max_blocks, output)
            },
        }
    }

    pub(super) fn execute_prepared(
        &mut self,
        input: &DeviceBuffer<bf16>,
        positions: &DeviceBuffer<u32>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        match self {
            Self::Linear(execution) => execution.execute_prepared_packed(input, output),
            Self::Full(execution) => execution.execute_prepared_packed(input, positions, output),
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

    pub(super) fn capture_partitions(&self) -> usize {
        match self {
            Self::Linear(_) => 0,
            Self::Full(execution) => execution.packed_capture_partitions(),
        }
    }
}
