use mircuda::{DeviceBuffer, bf16};
use runtime::kv::{BlockTable, KvWritePlan};

use super::{
    Configs, Dynamic, Geometry, Kernels, Nodes,
    arguments::{AttentionArguments, KvArguments},
};
use crate::{
    DecodeMoeBlockBf16, PagedPrefillBatch, PrefillMoeBlockBf16, Result,
    backend::block::graph::weights::CapturedBlockWeights,
    kernels::{
        MergeAttentionArguments, QkvPostprocessArguments, SplitAttentionArguments,
        SplitAttentionConfigs,
    },
};

pub(in crate::backend) struct CapturedLayer {
    pub(super) block: DecodeMoeBlockBf16,
    pub(super) input: DeviceBuffer<bf16>,
    pub(super) output: DeviceBuffer<bf16>,
    pub(super) weights: CapturedBlockWeights,
    write_plan: KvWritePlan,
    table: BlockTable,
    geometry: Geometry,
    dynamic: Dynamic,
    blocks: Vec<u32>,
}

impl CapturedLayer {
    pub(in crate::backend) fn new(
        mut block: DecodeMoeBlockBf16,
        input: DeviceBuffer<bf16>,
        weights: CapturedBlockWeights,
        write_plan: &KvWritePlan,
        table: &BlockTable,
        output: DeviceBuffer<bf16>,
    ) -> Result<Self> {
        let geometry = Geometry::new(&block.attention)?;
        let dynamic = Dynamic::new(write_plan, table, geometry)?;
        block.attention.prepare_capture_table(table)?;
        Ok(Self {
            block,
            input,
            output,
            weights,
            write_plan: write_plan.clone(),
            table: table.clone(),
            geometry,
            dynamic,
            blocks: table.blocks().iter().map(|block| block.0).collect(),
        })
    }

    pub(in crate::backend) fn capture(&mut self) -> Result<Nodes> {
        self.block.execute_captured(
            &self.input,
            self.weights.borrow(),
            &self.write_plan,
            &self.table,
            &mut self.output,
        )
    }

    pub(in crate::backend) fn prepare_replay(
        &mut self,
        write_plan: &KvWritePlan,
        table: &BlockTable,
    ) -> Result<()> {
        self.dynamic = Dynamic::new(write_plan, table, self.geometry)?;
        self.block.attention.prepare_capture_table(table)?;
        self.write_plan = write_plan.clone();
        self.table = table.clone();
        self.blocks.clear();
        self.blocks.extend(table.blocks().iter().map(|block| block.0));
        Ok(())
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
        prefill.execute_masked(
            &mut self.block,
            input,
            self.weights.borrow(),
            write_plan,
            table,
            start_position,
            output,
            image,
        )
    }

    pub(in crate::backend) fn execute_prefill_batch(
        &mut self,
        prefill: &mut PrefillMoeBlockBf16,
        input: &DeviceBuffer<bf16>,
        batch: &PagedPrefillBatch,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        prefill.execute_batch(&mut self.block, input, self.weights.borrow(), batch, output)
    }

    pub(in crate::backend) fn kernels(&self) -> Kernels {
        Kernels::new(&self.block.attention)
    }

    pub(in crate::backend) fn configs(&self) -> Result<Configs> {
        Configs::new(&self.block.attention)
    }

    pub(in crate::backend) const fn geometry(&self) -> Geometry {
        self.geometry
    }

    pub(in crate::backend) const fn set_dynamic(&mut self, dynamic: Dynamic) {
        self.dynamic = dynamic;
    }

    #[must_use]
    pub(in crate::backend) fn requires_recapture(&self, table: &BlockTable) -> bool {
        table.blocks().len() != self.blocks.len()
            || table
                .blocks()
                .iter()
                .zip(&self.blocks)
                .any(|(block, expected)| block.0 != *expected)
    }

    pub(in crate::backend) fn qkv_arguments(&mut self) -> QkvPostprocessArguments<'_> {
        let geometry = self.geometry;
        let dynamic = self.dynamic;
        let weights = self.weights.borrow();
        let scratch = &mut self.block.attention.scratch;
        (
            &scratch.qkv,
            &scratch.qkv,
            &scratch.qkv,
            weights.attention.query_norm,
            weights.attention.key_norm,
            &mut scratch.query_rope,
            &mut scratch.key_rope,
            &mut scratch.value_norm,
            1,
            geometry.query_heads,
            geometry.kv_heads,
            geometry.head_dim,
            geometry.value_head_dim,
            geometry.rotary_dim,
            geometry.pairing_dim,
            dynamic.position,
            geometry.theta,
            geometry.epsilon,
            0,
            u32::from(geometry.normalization.query),
            u32::from(geometry.normalization.key),
            u32::from(geometry.normalization.value),
        )
    }

    pub(in crate::backend) fn kv_arguments(&mut self) -> KvArguments<'_> {
        let geometry = self.geometry;
        let dynamic = self.dynamic;
        let attention = &mut self.block.attention;
        let (key_pages, value_pages) = attention.cache.pages_mut();
        (
            &attention.scratch.key_rope,
            &attention.scratch.value_norm,
            key_pages,
            value_pages,
            dynamic.local_start,
            1,
            dynamic.physical_block,
            dynamic.page_start,
            geometry.block_size_abi,
            geometry.kv_heads,
            geometry.head_dim,
            geometry.value_head_dim,
        )
    }

    pub(in crate::backend) fn attention_arguments(&mut self) -> AttentionArguments<'_> {
        let geometry = self.geometry;
        let dynamic = self.dynamic;
        let attention = &mut self.block.attention;
        (
            &attention.scratch.query_rope,
            attention.cache.key_pages(),
            attention.cache.value_pages(),
            attention.attention.table_device(),
            &mut attention.scratch.attention,
            dynamic.token_count,
            dynamic.block_count,
            geometry.block_size_abi,
            geometry.query_heads,
            geometry.kv_heads,
            geometry.head_dim,
            geometry.value_head_dim,
            geometry.window,
            geometry.scale,
            geometry.split_threshold,
        )
    }

    pub(in crate::backend) fn split_attention_configs(&self) -> Result<SplitAttentionConfigs> {
        self.block
            .attention
            .attention
            .captured_configs(self.dynamic.token_count, self.geometry.window)
    }

    pub(in crate::backend) fn split_attention_arguments(
        &mut self,
    ) -> Result<SplitAttentionArguments<'_>> {
        let geometry = self.geometry;
        let dynamic = self.dynamic;
        let attention = &mut self.block.attention;
        attention.attention.captured_split_arguments(
            &attention.scratch.query_rope,
            &attention.cache,
            dynamic.token_count,
            dynamic.block_count,
            geometry.window,
            geometry.scale,
        )
    }

    pub(in crate::backend) fn merge_attention_arguments(
        &mut self,
    ) -> Result<MergeAttentionArguments<'_>> {
        let geometry = self.geometry;
        let dynamic = self.dynamic;
        let attention = &mut self.block.attention;
        attention.attention.captured_merge_arguments(
            &mut attention.scratch.attention,
            dynamic.token_count,
            geometry.window,
        )
    }
}
