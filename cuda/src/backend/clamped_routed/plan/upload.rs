use runtime::kv::BlockTable;

use super::ClampedRoutedExecutionPlan;
use crate::{CudaClampedRoutedModelTemplate, Error, Result};

impl ClampedRoutedExecutionPlan {
    pub(in crate::backend::clamped_routed) fn upload(
        &mut self,
        template: &CudaClampedRoutedModelTemplate,
        tokens: &[u32],
        table: &BlockTable,
        start: usize,
        ring_slot: usize,
    ) -> Result<()> {
        if tokens.len() != self.tokens || table.blocks().len() > self.table_snapshot.len() {
            return Err(Error::InvalidPagedKv(
                "clamped-routed plan input differs from its geometry",
            ));
        }
        self.token_staging.copy_from_slice(tokens)?;
        let positions = (0..self.tokens)
            .map(|local| -> Result<u32> {
                Ok(u32::try_from(
                    start
                        .checked_add(local)
                        .ok_or(Error::InvalidPagedKv("clamped-routed position overflow"))?,
                )?)
            })
            .collect::<Result<Vec<_>>>()?;
        self.position_staging.copy_from_slice(&positions)?;
        self.table_snapshot.fill(u32::MAX);
        for (target, block) in self.table_snapshot.iter_mut().zip(table.blocks()) {
            *target = block.0;
        }
        self.ring_table_snapshot.fill(u32::MAX);
        if let Some(ring_blocks) = template.ring_blocks() {
            let base = ring_slot
                .checked_mul(ring_blocks)
                .ok_or(Error::InvalidPagedKv("windowed KV table offset overflow"))?;
            for (logical, target) in
                self.ring_table_snapshot.iter_mut().take(table.blocks().len()).enumerate()
            {
                *target = u32::try_from(base + logical % ring_blocks)?;
            }
        }
        self.table_staging.copy_from_slice(&self.table_snapshot)?;
        self.ring_table_staging.copy_from_slice(&self.ring_table_snapshot)?;
        let stream = &template.backend.inner.stream;
        stream.copy_to_device(&mut self.token_staging, &mut self.token_ids)?;
        stream.copy_to_device(&mut self.position_staging, &mut self.positions)?;
        stream.copy_to_device(&mut self.table_staging, &mut self.table_device)?;
        stream.copy_to_device(&mut self.ring_table_staging, &mut self.ring_table_device)?;
        Ok(())
    }

    pub(in crate::backend::clamped_routed) fn upload_packed(
        &mut self,
        template: &CudaClampedRoutedModelTemplate,
        tokens: &[u32],
    ) -> Result<()> {
        if tokens.len() != self.tokens {
            return Err(Error::InvalidPagedKv(
                "packed clamped-routed tokens differ from plan geometry",
            ));
        }
        self.token_staging.copy_from_slice(tokens)?;
        template
            .backend
            .inner
            .stream
            .copy_to_device(&mut self.token_staging, &mut self.token_ids)?;
        Ok(())
    }
}
