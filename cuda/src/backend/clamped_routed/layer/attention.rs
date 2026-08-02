use mircuda::{DeviceBuffer, bf16};
use runtime::kv::BlockTable;

use super::ClampedRoutedLayerExecution;
use crate::{PagedKvCache, Result, kernels::ClampedRoutedSplitDecode};

impl ClampedRoutedLayerExecution {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn attend(
        &self,
        cache: &PagedKvCache,
        current_keys: &DeviceBuffer<bf16>,
        current_values: &DeviceBuffer<bf16>,
        table: &BlockTable,
        table_device: &DeviceBuffer<u32>,
        sinks: &DeviceBuffer<bf16>,
        start: usize,
        query: &DeviceBuffer<bf16>,
        attended: &mut DeviceBuffer<bf16>,
        split_decode: &mut Option<ClampedRoutedSplitDecode>,
    ) -> Result<()> {
        let blocks = table.blocks().len();
        if self.tokens == 1 {
            if !cache.is_windowed()
                && let Some(split) = split_decode.as_mut()
            {
                split.ensure_tuned(
                    &self.attention,
                    &self.stream,
                    query,
                    cache.key_pages(),
                    cache.value_pages(),
                    table_device,
                    sinks,
                    attended,
                    table.token_len(),
                    blocks,
                    self.window,
                    self.config.scale,
                );
                if split.execute(
                    &self.stream,
                    query,
                    cache.key_pages(),
                    cache.value_pages(),
                    table_device,
                    sinks,
                    attended,
                    table.token_len(),
                    blocks,
                    self.window,
                    self.config.scale,
                )? {
                    return Ok(());
                }
            }
            self.attention.execute(
                &self.stream,
                query,
                current_keys,
                current_values,
                cache.key_pages(),
                cache.value_pages(),
                table_device,
                sinks,
                attended,
                table.token_len(),
                blocks,
                self.window,
                self.config.scale,
            )
        } else {
            self.attention.execute_prefill(
                &self.stream,
                query,
                current_keys,
                current_values,
                cache.key_pages(),
                cache.value_pages(),
                table_device,
                sinks,
                attended,
                self.tokens,
                start,
                blocks,
                self.window,
                self.config.scale,
            )
        }
    }
}
