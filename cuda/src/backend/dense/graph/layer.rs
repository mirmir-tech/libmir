use mircuda::{DeviceBuffer, bf16};
use runtime::kv::{BlockTable, KvWritePlan};

use super::{
    arguments::{AttentionArguments, KvArguments},
    weights::CapturedDenseWeights,
};
use crate::{
    DecodeDenseSwiGlu, DenseSwiGluWeights, PrefillDenseSwiGlu, Result,
    backend::attention::graph::{Configs, Dynamic, Geometry, Kernels, Nodes},
    kernels::{
        MergeAttentionArguments, QkvPostprocessArguments, SplitAttentionArguments,
        SplitAttentionConfigs,
    },
};

pub(in crate::backend) struct PreparedDecodeDense {
    pub block: DecodeDenseSwiGlu,
    pub input: DeviceBuffer<bf16>,
    pub output: DeviceBuffer<bf16>,
    pub weights: DenseSwiGluWeightsOwned,
}

pub(in crate::backend) struct DenseSwiGluWeightsOwned(CapturedDenseWeights);

impl TryFrom<DenseSwiGluWeights<'_>> for DenseSwiGluWeightsOwned {
    type Error = crate::Error;

    fn try_from(weights: DenseSwiGluWeights<'_>) -> Result<Self> {
        Ok(Self(weights.try_into()?))
    }
}

pub(in crate::backend) struct CapturedDenseLayer {
    block: DecodeDenseSwiGlu,
    input: DeviceBuffer<bf16>,
    output: DeviceBuffer<bf16>,
    weights: CapturedDenseWeights,
    write_plan: KvWritePlan,
    table: BlockTable,
    geometry: Geometry,
    dynamic: Dynamic,
}

impl CapturedDenseLayer {
    pub(in crate::backend) fn new(
        mut prepared: PreparedDecodeDense,
        write_plan: &KvWritePlan,
        table: &BlockTable,
    ) -> Result<Self> {
        let geometry = Geometry::new(&prepared.block.attention)?;
        let dynamic = Dynamic::new(write_plan, table, geometry)?;
        prepared.block.attention.prepare_capture_table(table)?;
        Ok(Self {
            block: prepared.block,
            input: prepared.input,
            output: prepared.output,
            weights: prepared.weights.0,
            write_plan: write_plan.clone(),
            table: table.clone(),
            geometry,
            dynamic,
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
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::backend) fn execute_prefill(
        &mut self,
        prefill: &mut PrefillDenseSwiGlu,
        input: &DeviceBuffer<bf16>,
        write_plan: &KvWritePlan,
        table: &BlockTable,
        start_position: usize,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        prefill.execute(
            &mut self.block,
            input,
            self.weights.borrow(),
            write_plan,
            table,
            start_position,
            output,
        )
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

    pub(in crate::backend) fn qkv_arguments(&mut self) -> QkvPostprocessArguments<'_> {
        let geometry = self.geometry;
        let dynamic = self.dynamic;
        let weights = self.weights.borrow();
        let scratch = &mut self.block.attention.scratch;
        let (query, key, value) = if geometry.separate_qkv {
            (&scratch.qkv_separate[0], &scratch.qkv_separate[1], &scratch.qkv_separate[2])
        } else {
            (&scratch.qkv, &scratch.qkv, &scratch.qkv)
        };
        (
            query,
            key,
            value,
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
            u32::from(geometry.separate_qkv),
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
