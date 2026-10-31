//! One layer of a scalar execution plan: dispatch over the layer kinds.

use mircuda::{DeviceBuffer, bf16};
use runtime::kv::{BlockTable, KvWritePlan};
use uuid::Uuid;

use super::super::{CudaSharedRoutedLayerState, SharedRoutedLayerTemplate};
use crate::{
    CudaAffineGatedDeltaMoeExecution, CudaAffineGatedFullAttentionMoeExecution,
    CudaAttentionHybridExecution, CudaMamba2HybridExecution, Error, Result,
};

#[derive(Debug)]
pub(super) enum SharedRoutedLayerExecution {
    Linear(Box<CudaAffineGatedDeltaMoeExecution>),
    Full(Box<CudaAffineGatedFullAttentionMoeExecution>),
    Mamba2(Box<CudaMamba2HybridExecution>),
    Attention(Box<CudaAttentionHybridExecution>),
}

/// The sequence-level context of one plan pass.
pub(super) struct PlanStep<'a> {
    pub(super) session_id: Uuid,
    pub(super) table: &'a BlockTable,
    pub(super) start_position: usize,
    pub(super) tokens: usize,
    pub(super) positions: &'a DeviceBuffer<u32>,
    pub(super) image_span: Option<(usize, usize)>,
}

impl SharedRoutedLayerExecution {
    pub(super) fn new(layer: &SharedRoutedLayerTemplate, tokens: usize) -> Result<Self> {
        Ok(match layer {
            SharedRoutedLayerTemplate::Linear(layer) => {
                Self::Linear(Box::new(layer.prepare(tokens)?))
            },
            SharedRoutedLayerTemplate::Full(layer) => Self::Full(Box::new(layer.prepare(tokens)?)),
            SharedRoutedLayerTemplate::Mamba2(layer) => {
                Self::Mamba2(Box::new(layer.prepare(tokens)?))
            },
            SharedRoutedLayerTemplate::Attention(layer) => {
                Self::Attention(Box::new(layer.prepare(tokens)?))
            },
        })
    }

    pub(super) fn execute(
        &mut self,
        index: usize,
        state: &mut CudaSharedRoutedLayerState,
        step: &PlanStep<'_>,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let write = || {
            KvWritePlan::prefill(
                step.session_id,
                index,
                step.table,
                step.start_position,
                step.tokens,
            )
        };
        match (self, state) {
            (Self::Linear(layer), CudaSharedRoutedLayerState::Linear(state)) => {
                if state.offset() != step.start_position {
                    return Err(Error::InvalidDecoderKernel(
                        "Gated Delta session position mismatch",
                    ));
                }
                layer.execute(input, state, output)
            },
            (Self::Full(layer), CudaSharedRoutedLayerState::Full(state)) => layer
                .execute_with_image_span(
                    input,
                    step.positions,
                    state,
                    &write()?,
                    step.table,
                    step.start_position,
                    None,
                    step.image_span,
                    output,
                ),
            (Self::Mamba2(layer), CudaSharedRoutedLayerState::Mamba2(state)) => {
                if state.offset() != step.start_position {
                    return Err(Error::InvalidDecoderKernel("Mamba-2 session position mismatch"));
                }
                layer.execute(input, state, output)
            },
            (Self::Attention(layer), CudaSharedRoutedLayerState::Attention(state)) => {
                layer.execute(input, state, &write()?, step.table, step.start_position, output)
            },
            _ => Err(Error::InvalidDecoderKernel("shared-routed execution and state kinds differ")),
        }
    }
}
