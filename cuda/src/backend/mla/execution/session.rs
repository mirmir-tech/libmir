//! Single-session execution: one sequence's rows at consecutive positions,
//! its block table uploaded before the pass.

use mircuda::{DeviceBuffer, PinnedBuffer, bf16};
use runtime::kv::BlockTable;

use super::{CudaLatentAttentionExecution, Placement};
use crate::{
    Error, LatentAttentionState, Result,
    kernels::{LatentExpandedRow, LatentExpandedRows},
};

/// Device block table and row slots of one sequence, and the all-zero
/// sequence index of every row.
#[derive(Debug)]
pub(super) struct SessionPaging {
    staging: PinnedBuffer<u32>,
    table: DeviceBuffer<u32>,
    slot_staging: PinnedBuffer<u32>,
    slot_mapping: DeviceBuffer<u32>,
    sequences: DeviceBuffer<u32>,
}

impl CudaLatentAttentionExecution {
    /// Advances one session by this execution's rows, at `positions` from
    /// `start`, which its `table` already covers.
    pub(crate) fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        state: &LatentAttentionState,
        table: &BlockTable,
        (start, positions): (usize, &DeviceBuffer<u32>),
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let blocks = table.blocks();
        if blocks.is_empty() {
            return Err(Error::InvalidPagedKv("latent session has no cache blocks"));
        }
        let rows = self.absorption.tokens();
        if self.session.as_ref().is_none_or(|paging| paging.table.len() < blocks.len()) {
            let capacity = blocks.len().next_power_of_two();
            let backend = &self.backend.inner;
            self.session = Some(SessionPaging {
                staging: backend.context.allocate_pinned(capacity)?,
                table: backend.pool.allocate(&backend.stream, capacity)?,
                slot_staging: backend.context.allocate_pinned(rows)?,
                slot_mapping: backend.pool.allocate(&backend.stream, rows)?,
                sequences: backend.pool.allocate_zeroed(&backend.stream, rows)?,
            });
        }
        let paging = self
            .session
            .take()
            .ok_or(Error::InvalidDecoderKernel("latent session paging is missing"))?;
        let block_size = self.layer.block_size;
        let slots = (start..start + rows)
            .map(|position| {
                let block = blocks
                    .get(position / block_size)
                    .ok_or(Error::InvalidPagedKv("latent session table misses a row"))?;
                Ok(block.0 * u32::try_from(block_size)? + u32::try_from(position % block_size)?)
            })
            .collect::<Result<Vec<_>>>()?;
        self.execute_session(input, state, (blocks, &slots), (start, positions), paging, output)
    }

    fn execute_session(
        &mut self,
        input: &DeviceBuffer<bf16>,
        state: &LatentAttentionState,
        (blocks, slots): (&[runtime::kv::BlockId], &[u32]),
        (start, positions): (usize, &DeviceBuffer<u32>),
        mut paging: SessionPaging,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let mut host = vec![0; paging.table.len()];
        for (target, block) in host.iter_mut().zip(blocks) {
            *target = block.0;
        }
        paging.staging.copy_from_slice(&host)?;
        paging.slot_staging.copy_from_slice(slots)?;
        let stream = &self.backend.inner.stream;
        stream.copy_to_device(&mut paging.staging, &mut paging.table)?;
        stream.copy_to_device(&mut paging.slot_staging, &mut paging.slot_mapping)?;
        let placement = Placement {
            positions,
            slot_mapping: &paging.slot_mapping,
            sequences: &paging.sequences,
            tables: &paging.table,
            table_stride: paging.table.len(),
            expanded: Some(LatentExpandedRows::One(LatentExpandedRow {
                start,
                tokens: slots.len(),
            })),
        };
        let mut pages = state.cache.clone();
        let result = self.run(input, &mut pages, &placement, output);
        self.session = Some(paging);
        result
    }
}
