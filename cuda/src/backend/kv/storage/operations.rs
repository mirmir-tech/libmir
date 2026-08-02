use mircuda::{DeviceBuffer, KernelNode, bf16};
use runtime::kv::KvWritePlan;

use super::PagedKvCache;
use crate::{Error, PagedDecodeBatch, PagedPrefillBatch, Result, kernels::KvStoreKernel};

impl PagedKvCache {
    pub(crate) fn copy_ring_slot(&mut self, source_slot: usize, target_slot: usize) -> Result<()> {
        let ring = self.ring.ok_or(Error::InvalidPagedKv("cannot copy a non-windowed KV ring"))?;
        let physical_blocks = ring.physical_blocks()?;
        let source_blocks = ring.slot_blocks(source_slot)?;
        let target_blocks = ring.slot_blocks(target_slot)?;
        copy_slot(
            &self.stream,
            &mut self.key_pages,
            physical_blocks,
            source_blocks.clone(),
            target_blocks.start,
        )?;
        copy_slot(
            &self.stream,
            &mut self.value_pages,
            physical_blocks,
            source_blocks,
            target_blocks.start,
        )
    }

    pub(crate) fn store_captured(
        &mut self,
        plan: &KvWritePlan,
        keys: &DeviceBuffer<bf16>,
        values: &DeviceBuffer<bf16>,
    ) -> Result<KernelNode<KvStoreKernel>> {
        self.validate_plan(plan)?;
        let [write] = plan.writes() else {
            return Err(Error::InvalidPagedKv("captured decode requires one KV page write"));
        };
        self.operation.execute_captured(
            &self.stream,
            keys,
            values,
            &mut self.key_pages,
            &mut self.value_pages,
            write.local_start,
            write.token_count(),
            usize::try_from(write.page.block.0)?,
            write.page_start,
        )
    }

    pub(crate) fn store_batch(
        &mut self,
        batch: &PagedDecodeBatch,
        keys: &DeviceBuffer<bf16>,
        values: &DeviceBuffer<bf16>,
    ) -> Result<()> {
        if batch.cache_config() != self.storage.cache || batch.active() == 0 {
            return Err(Error::InvalidPagedKv("batched KV metadata geometry differs"));
        }
        self.operation.execute_batch(
            &self.stream,
            keys,
            values,
            &mut self.key_pages,
            &mut self.value_pages,
            batch.tables(),
            batch.token_counts(),
            batch.active(),
            batch.max_blocks(),
        )
    }

    pub fn store_prefill_batch(
        &mut self,
        batch: &PagedPrefillBatch,
        keys: &DeviceBuffer<bf16>,
        values: &DeviceBuffer<bf16>,
    ) -> Result<()> {
        if batch.cache_config() != self.storage.cache || batch.tokens() == 0 {
            return Err(Error::InvalidPagedKv("prefill KV metadata geometry differs"));
        }
        let windowed = self.is_windowed();
        self.operation.execute_prefill_batch(
            &self.stream,
            keys,
            values,
            &mut self.key_pages,
            &mut self.value_pages,
            batch.slot_mapping(windowed),
            batch.tokens(),
        )
    }

    pub(crate) fn store_for_session(
        &mut self,
        plan: &KvWritePlan,
        session_slot: usize,
        keys: &DeviceBuffer<bf16>,
        values: &DeviceBuffer<bf16>,
    ) -> Result<usize> {
        self.validate_plan(plan)?;
        for write in plan.writes() {
            let block = match self.ring {
                Some(ring) => ring.write_block(session_slot, *write)?,
                None => usize::try_from(write.page.block.0)?,
            };
            self.operation.execute(
                &self.stream,
                keys,
                values,
                &mut self.key_pages,
                &mut self.value_pages,
                write.local_start,
                write.token_count(),
                block,
                write.page_start,
            )?;
        }
        Ok(plan.written_tokens())
    }
}

fn copy_slot(
    stream: &mircuda::Stream,
    pages: &mut DeviceBuffer<u8>,
    physical_blocks: usize,
    source_blocks: std::ops::Range<usize>,
    target_block: usize,
) -> Result<()> {
    if physical_blocks == 0 || !pages.len().is_multiple_of(physical_blocks) {
        return Err(Error::InvalidPagedKv("windowed KV page geometry is not divisible"));
    }
    let bytes_per_block = pages.len() / physical_blocks;
    let source = source_blocks.start * bytes_per_block..source_blocks.end * bytes_per_block;
    let target = target_block * bytes_per_block;
    let alias = pages.clone();
    Ok(stream.copy_device_range(&alias, source, pages, target)?)
}
