use mircuda::{DeviceBuffer, KernelNode, LaunchConfig, Stream, TypedKernel, bf16};

use super::{
    MergeAttentionKernel, SplitAttentionKernel, SplitAttentionWorkspace, SplitPagedAttention,
};
use crate::{
    Result,
    kernels::geometry::{narrow, product},
};

pub type SplitAttentionArguments<'a> = (
    &'a DeviceBuffer<bf16>,
    &'a DeviceBuffer<u8>,
    &'a DeviceBuffer<u8>,
    &'a DeviceBuffer<u32>,
    &'a mut DeviceBuffer<f32>,
    &'a mut DeviceBuffer<f32>,
    &'a mut DeviceBuffer<f32>,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    f32,
    u32,
    u32,
    u32,
    u32,
);

pub type MergeAttentionArguments<'a> = (
    &'a DeviceBuffer<f32>,
    &'a DeviceBuffer<f32>,
    &'a DeviceBuffer<f32>,
    &'a mut DeviceBuffer<bf16>,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
);

#[derive(Clone, Copy)]
pub struct SplitAttentionNodes {
    pub(crate) split: KernelNode<SplitAttentionKernel>,
    pub(crate) merge: KernelNode<MergeAttentionKernel>,
}

#[derive(Clone)]
pub struct SplitAttentionKernels {
    pub(crate) split: TypedKernel<SplitAttentionKernel>,
    pub(crate) merge: TypedKernel<MergeAttentionKernel>,
}

#[derive(Clone, Copy, Debug)]
pub struct SplitAttentionConfigs {
    pub(crate) split: LaunchConfig,
    pub(crate) merge: LaunchConfig,
}

impl SplitPagedAttention {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_captured(
        &self,
        stream: &Stream,
        query: &DeviceBuffer<bf16>,
        key_pages: &DeviceBuffer<u8>,
        value_pages: &DeviceBuffer<u8>,
        block_table: &DeviceBuffer<u32>,
        workspace: &mut SplitAttentionWorkspace,
        output: &mut DeviceBuffer<bf16>,
        token_count: usize,
        block_count: usize,
        window: Option<usize>,
        scale: f32,
        minimum_tokens: usize,
    ) -> Result<SplitAttentionNodes> {
        let active = self.validate(
            query, block_table, workspace, output, token_count, block_count, window, scale,
        )?;
        let configs = self.configs(active)?;
        let split = self.split.launch_captured(
            stream,
            configs.split,
            self.split_arguments(
                query, key_pages, value_pages, block_table, workspace, token_count, block_count,
                window, scale, active, minimum_tokens,
            )?,
        )?;
        let merge = self.merge.launch_captured(
            stream,
            configs.merge,
            self.merge_arguments(
                workspace,
                output,
                window.map_or(token_count, |limit| token_count.min(limit)),
                active,
                minimum_tokens,
            )?,
        )?;
        Ok(SplitAttentionNodes { split, merge })
    }

    pub(crate) fn kernels(&self) -> SplitAttentionKernels {
        SplitAttentionKernels {
            split: self.split.clone(),
            merge: self.merge.clone(),
        }
    }

    pub(crate) fn configs(&self, active: usize) -> Result<SplitAttentionConfigs> {
        Ok(SplitAttentionConfigs {
            split: LaunchConfig {
                grid: (narrow(product(self.split_heads(), active)?)?, 1, 1),
                block: (self.threads(), 1, 1),
                shared_memory_bytes: 0,
            },
            merge: LaunchConfig {
                grid: (narrow(self.spec.query_heads)?, 1, 1),
                block: (self.threads(), 1, 1),
                shared_memory_bytes: narrow(active * size_of::<f32>())?,
            },
        })
    }

    pub(crate) fn active_partitions(&self, token_count: u32, window: u32) -> Result<u32> {
        let visible = if window == 0 {
            token_count
        } else {
            token_count.min(window)
        };
        Ok(visible.div_ceil(u32::try_from(self.partition_tokens)?))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn split_arguments<'a>(
        &self,
        query: &'a DeviceBuffer<bf16>,
        key_pages: &'a DeviceBuffer<u8>,
        value_pages: &'a DeviceBuffer<u8>,
        block_table: &'a DeviceBuffer<u32>,
        workspace: &'a mut SplitAttentionWorkspace,
        token_count: usize,
        block_count: usize,
        window: Option<usize>,
        scale: f32,
        active: usize,
        minimum_tokens: usize,
    ) -> Result<SplitAttentionArguments<'a>> {
        Ok((
            query,
            key_pages,
            value_pages,
            block_table,
            &mut workspace.values,
            &mut workspace.maxima,
            &mut workspace.denominators,
            narrow(token_count)?,
            narrow(block_count)?,
            narrow(self.spec.block_size)?,
            narrow(self.spec.query_heads)?,
            narrow(self.spec.kv_heads)?,
            narrow(self.spec.head_dim)?,
            narrow(self.spec.value_head_dim)?,
            narrow(window.unwrap_or(0))?,
            scale,
            narrow(self.partition_tokens)?,
            narrow(active)?,
            narrow(self.max_partitions)?,
            narrow(minimum_tokens)?,
        ))
    }

    pub(crate) fn merge_arguments<'a>(
        &self,
        workspace: &'a SplitAttentionWorkspace,
        output: &'a mut DeviceBuffer<bf16>,
        visible_tokens: usize,
        active: usize,
        minimum_tokens: usize,
    ) -> Result<MergeAttentionArguments<'a>> {
        Ok((
            &workspace.values,
            &workspace.maxima,
            &workspace.denominators,
            output,
            narrow(self.spec.query_heads)?,
            narrow(self.spec.value_head_dim)?,
            narrow(active)?,
            narrow(self.max_partitions)?,
            narrow(visible_tokens)?,
            narrow(minimum_tokens)?,
        ))
    }
}
