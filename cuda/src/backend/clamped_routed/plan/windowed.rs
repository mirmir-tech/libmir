use super::ClampedRoutedExecutionPlan;
use crate::{
    CudaClampedRoutedModelTemplate, PagedPrefillBatch, Result, backend::WindowedPrefillStaging,
};

impl ClampedRoutedExecutionPlan {
    pub(super) fn prepare_windowed_prefill(
        &mut self,
        template: &CudaClampedRoutedModelTemplate,
        batch: &PagedPrefillBatch,
    ) -> Result<()> {
        let Some(window) = template
            .max_sliding_window()
            .filter(|_| batch.max_query_tokens() >= super::super::WINDOWED_FMHA_MIN_QUERY_TOKENS)
        else {
            return Ok(());
        };
        let storage = template.config.storage(template.cache);
        let matches = self.windowed_prefill.as_ref().is_some_and(|staging| {
            staging.matches(batch.row_capacity(), batch.token_capacity(), storage, window)
        });
        if !matches {
            self.windowed_prefill = Some(WindowedPrefillStaging::new(
                &template.backend,
                storage,
                batch.row_capacity(),
                batch.token_capacity(),
                window,
            )?);
        }
        Ok(())
    }
}
