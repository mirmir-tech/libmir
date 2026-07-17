use super::{
    BlockTable, Configs, DecodeAttentionBf16, DecodeAttentionWeights, DeviceBuffer, Dynamic, Error,
    Geometry, Kernels, KvWritePlan, LaunchConfig, Nodes, Result, bf16,
};
use crate::backend::attention::qkv::QkvProjectionBuffers;

impl Dynamic {
    pub(in crate::backend) fn new(
        plan: &KvWritePlan,
        table: &BlockTable,
        geometry: Geometry,
    ) -> Result<Self> {
        let [write] = plan.writes() else {
            return Err(Error::InvalidPagedKv("captured decode requires one KV page write"));
        };
        if plan.token_count() != 1
            || write.token_count() != 1
            || write.page.layer != geometry.layer
            || plan.block_size() != geometry.block_size
            || table.block_size() != Some(geometry.block_size)
            || table.token_len() == 0
            || write.page.block.0 >= geometry.block_count
            || write.page_start >= geometry.block_size
            || table.token_len() > table.blocks().len().saturating_mul(geometry.block_size)
        {
            return Err(Error::InvalidPagedKv("invalid captured decode KV step"));
        }
        Ok(Self {
            position: u32::try_from(table.token_len() - 1)?,
            local_start: u32::try_from(write.local_start)?,
            physical_block: write.page.block.0,
            page_start: u32::try_from(write.page_start)?,
            token_count: u32::try_from(table.token_len())?,
            block_count: u32::try_from(table.blocks().len())?,
        })
    }
}

impl Geometry {
    pub(in crate::backend) fn new(attention: &DecodeAttentionBf16) -> Result<Self> {
        let config = attention.config;
        Ok(Self {
            layer: config.layer,
            block_size: config.cache.cache.block_size,
            block_size_abi: u32::try_from(config.cache.cache.block_size)?,
            block_count: config.cache.cache.block_count,
            query_heads: u32::try_from(config.query_heads)?,
            kv_heads: u32::try_from(config.cache.kv_heads)?,
            head_dim: u32::try_from(config.cache.key_head_dim)?,
            value_head_dim: u32::try_from(config.cache.value_head_dim)?,
            rotary_dim: u32::try_from(config.rotary_dim)?,
            pairing_dim: u32::try_from(config.rope_pairing_dim)?,
            window: u32::try_from(config.sliding_window.unwrap_or(0))?,
            scale: config.attention_scale,
            split_threshold: attention.attention.split_threshold(),
            theta: config.rope_theta,
            epsilon: config.rms_norm_epsilon,
            normalization: config.qkv_normalization,
            separate_qkv: config.projection_format == crate::ProjectionFormat::NvFp4,
        })
    }
}

impl Kernels {
    pub(in crate::backend) fn new(attention: &DecodeAttentionBf16) -> Self {
        Self {
            qkv_postprocess: attention.qkv_postprocess.kernel(),
            kv_store: attention.cache.kernel(),
            attention: attention.attention.captured_kernels(),
        }
    }
}

impl Configs {
    pub(in crate::backend) fn new(attention: &DecodeAttentionBf16) -> Result<Self> {
        let geometry = Geometry::new(attention)?;
        let width = geometry
            .kv_heads
            .checked_mul(geometry.head_dim.max(geometry.value_head_dim))
            .ok_or(Error::InvalidPagedKv("KV launch overflow"))?;
        Ok(Self {
            qkv_postprocess: attention.qkv_postprocess.config()?,
            kv_store: LaunchConfig {
                grid: (width.div_ceil(256), 1, 1),
                block: (256, 1, 1),
                shared_memory_bytes: 0,
            },
            attention: LaunchConfig {
                grid: (geometry.query_heads, 1, 1),
                block: (256, 1, 1),
                shared_memory_bytes: 0,
            },
        })
    }
}

impl DecodeAttentionBf16 {
    pub(in crate::backend) fn execute_captured(
        &mut self,
        input: &DeviceBuffer<bf16>,
        weights: DecodeAttentionWeights<'_>,
        write_plan: &KvWritePlan,
        table: &BlockTable,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<Nodes> {
        let position = table.token_len() - 1;
        let separate = self.qkv.execute(
            input,
            &self.input_norm,
            weights.input_norm,
            weights.qkv,
            &mut QkvProjectionBuffers {
                normalized: &mut self.scratch.normalized,
                packed: &mut self.scratch.qkv,
                separate: &mut self.scratch.qkv_separate,
            },
        )?;
        let qkv_postprocess = if separate {
            self.qkv_postprocess.execute_captured_separate(
                &self.stream,
                [
                    &self.scratch.qkv_separate[0],
                    &self.scratch.qkv_separate[1],
                    &self.scratch.qkv_separate[2],
                ],
                weights.query_norm,
                weights.key_norm,
                &mut self.scratch.query_rope,
                &mut self.scratch.key_rope,
                &mut self.scratch.value_norm,
                position,
            )?
        } else {
            self.qkv_postprocess.execute_captured(
                &self.stream,
                &self.scratch.qkv,
                weights.query_norm,
                weights.key_norm,
                &mut self.scratch.query_rope,
                &mut self.scratch.key_rope,
                &mut self.scratch.value_norm,
                position,
            )?
        };
        let kv_store = self.cache.store_captured(
            write_plan,
            &self.scratch.key_rope,
            &self.scratch.value_norm,
        )?;
        let attention = self.attention.execute_captured(
            &self.scratch.query_rope,
            &self.cache,
            table,
            &mut self.scratch.attention,
            self.config.sliding_window,
            self.config.attention_scale,
        )?;
        self.output_projection.execute(
            &self.stream,
            &self.scratch.attention,
            weights.output,
            output,
        )?;
        Ok(Nodes { qkv_postprocess, kv_store, attention })
    }

    pub(in crate::backend) fn prepare_capture_table(&mut self, table: &BlockTable) -> Result<()> {
        self.attention.prepare_table(table)
    }
}
