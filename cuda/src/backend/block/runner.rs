use ::runtime::kv::{BlockTable, KvWritePlan};
use mircuda::{DeviceBuffer, bf16};

use super::{
    CapturedDecodeMoeBlockBf16, DecodeMoeBlockBf16, PrefillMoeBlockBf16,
    graph::{capture_block, weights::CapturedBlockWeights},
};
use crate::{Error, Result};

/// Work performed by one session-local decoder-layer step.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecodeGraphAction {
    /// The token ran directly and initialized a reusable graph.
    CapturedAfterDirect,
    /// The token replayed the existing graph after typed scalar rebinding.
    Replayed,
    /// The token replayed a replacement graph after a K/V mapping change.
    Recaptured,
}

/// Session-local owner of one prepared or captured routed-MoE layer.
pub struct DecodeMoeBlockExecutor {
    state: Option<State>,
    input: DeviceBuffer<bf16>,
    output: DeviceBuffer<bf16>,
    layer: usize,
}

pub(in crate::backend) struct PreparedDecodeMoeBlock {
    pub(in crate::backend) layer: usize,
    pub(in crate::backend) block: DecodeMoeBlockBf16,
    pub(in crate::backend) weights: CapturedBlockWeights,
    pub(in crate::backend) input: DeviceBuffer<bf16>,
    pub(in crate::backend) output: DeviceBuffer<bf16>,
}

enum State {
    Prepared {
        block: Box<DecodeMoeBlockBf16>,
        weights: Box<CapturedBlockWeights>,
    },
    Captured(Box<CapturedDecodeMoeBlockBf16>),
}

impl DecodeMoeBlockExecutor {
    pub(super) fn new(
        block: DecodeMoeBlockBf16,
        input: DeviceBuffer<bf16>,
        weights: CapturedBlockWeights,
        output: DeviceBuffer<bf16>,
    ) -> Self {
        let layer = block.config.attention.layer;
        Self {
            state: Some(State::Prepared {
                block: Box::new(block),
                weights: Box::new(weights),
            }),
            input,
            output,
            layer,
        }
    }

    /// Shared handle to the fixed graph input allocation.
    #[must_use]
    pub const fn input(&self) -> &DeviceBuffer<bf16> {
        &self.input
    }

    /// Shared handle to the fixed graph output allocation.
    #[must_use]
    pub const fn output(&self) -> &DeviceBuffer<bf16> {
        &self.output
    }

    pub(in crate::backend) fn execute_direct(
        &mut self,
        write_plan: &KvWritePlan,
        table: &BlockTable,
    ) -> Result<()> {
        let state = self
            .state
            .as_mut()
            .ok_or(Error::InvalidDecoderKernel("CUDA block executor is unavailable"))?;
        match state {
            State::Prepared { block, weights } => {
                block.execute(&self.input, weights.borrow(), write_plan, table, &mut self.output)
            },
            State::Captured(_) => {
                Err(Error::InvalidDecoderKernel("captured CUDA block cannot join a model graph"))
            },
        }
    }

    pub(in crate::backend) fn into_prepared(mut self) -> Result<PreparedDecodeMoeBlock> {
        let state = self
            .state
            .take()
            .ok_or(Error::InvalidDecoderKernel("CUDA block executor is unavailable"))?;
        match state {
            State::Prepared { block, weights } => Ok(PreparedDecodeMoeBlock {
                layer: self.layer,
                block: *block,
                weights: *weights,
                input: self.input,
                output: self.output,
            }),
            State::Captured(_) => {
                Err(Error::InvalidDecoderKernel("captured CUDA block cannot join a model graph"))
            },
        }
    }

    /// Executes one token and maintains the appropriate CUDA Graph capture.
    pub fn execute(
        &mut self,
        write_plan: &KvWritePlan,
        table: &BlockTable,
    ) -> Result<DecodeGraphAction> {
        let state = self
            .state
            .take()
            .ok_or(Error::InvalidDecoderKernel("CUDA block executor is unavailable"))?;
        let (state, action) = self.execute_state(state, write_plan, table)?;
        self.state = Some(state);
        tracing::debug!(
            layer = self.layer,
            token = table.token_len(),
            blocks = table.blocks().len(),
            action = ?action,
            "executed CUDA MoE block"
        );
        Ok(action)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute_prefill(
        &mut self,
        prefill: &mut PrefillMoeBlockBf16,
        input: &DeviceBuffer<bf16>,
        write_plan: &KvWritePlan,
        table: &BlockTable,
        start_position: usize,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.execute_prefill_masked(prefill, input, write_plan, table, start_position, output, None)
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::backend) fn execute_prefill_masked(
        &mut self,
        prefill: &mut PrefillMoeBlockBf16,
        input: &DeviceBuffer<bf16>,
        write_plan: &KvWritePlan,
        table: &BlockTable,
        start_position: usize,
        output: &mut DeviceBuffer<bf16>,
        image: Option<crate::backend::attention::ImageAttentionSpan>,
    ) -> Result<()> {
        let state = self
            .state
            .as_mut()
            .ok_or(Error::InvalidDecoderKernel("CUDA block executor is unavailable"))?;
        match state {
            State::Prepared { block, weights } => prefill.execute_masked(
                block,
                input,
                weights.borrow(),
                write_plan,
                table,
                start_position,
                output,
                image,
            ),
            State::Captured(_) => Err(Error::InvalidDecoderKernel(
                "CUDA prefill must complete before decode graph capture",
            )),
        }
    }

    fn execute_state(
        &mut self,
        state: State,
        write_plan: &KvWritePlan,
        table: &BlockTable,
    ) -> Result<(State, DecodeGraphAction)> {
        match state {
            State::Prepared { mut block, weights } => {
                block.execute(
                    &self.input,
                    weights.borrow(),
                    write_plan,
                    table,
                    &mut self.output,
                )?;
                let graph = capture_block(
                    *block,
                    self.input.clone(),
                    *weights,
                    write_plan,
                    table,
                    self.output.clone(),
                )?;
                Ok((State::Captured(Box::new(graph)), DecodeGraphAction::CapturedAfterDirect))
            },
            State::Captured(mut graph) if !graph.requires_recapture(table) => {
                graph.execute(write_plan, table)?;
                Ok((State::Captured(graph), DecodeGraphAction::Replayed))
            },
            State::Captured(graph) => {
                let mut graph = (*graph).recapture(write_plan, table)?;
                graph.launch()?;
                Ok((State::Captured(Box::new(graph)), DecodeGraphAction::Recaptured))
            },
        }
    }
}
