use mircuda::{DeviceBuffer, bf16};
use runtime::{
    backend::SamplingLogits,
    kv::{BlockTable, KvWritePlan},
};
use uuid::Uuid;

use super::CudaMoeModelSession;
use crate::{
    Error, Result, backend::attention::ImageAttentionSpan, kernels::VisionEmbeddingSplice,
};

impl CudaMoeModelSession {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prefill_vision_for_sampling(
        &mut self,
        session_id: Uuid,
        tokens: &[u32],
        image: &DeviceBuffer<bf16>,
        image_start: usize,
        image_end: usize,
        bidirectional: bool,
        table: &BlockTable,
        sampling: SamplingLogits,
    ) -> Result<&DeviceBuffer<bf16>> {
        self.validate_prefill(tokens, 0, table)?;
        if image_start >= image_end
            || image_end > tokens.len()
            || image.len() != (image_end - image_start) * self.hidden_size
        {
            return Err(Error::InvalidVisionKernel("image embeddings differ from prompt span"));
        }
        self.ensure_prefill_capacity(tokens.len())?;
        self.prefill_tokens.upload(&self.stream, tokens)?;
        self.embedding.execute_batch(
            self.prefill_tokens.device(),
            0,
            tokens.len(),
            &mut self.prefill_first,
        )?;
        VisionEmbeddingSplice::compile(
            &self.backend.inner.compiler,
            image_end - image_start,
            self.hidden_size,
            image_start,
        )?
        .execute(&self.stream, image, &mut self.prefill_first)?;
        let image =
            bidirectional.then_some(ImageAttentionSpan { start: image_start, end: image_end });
        self.execute_vision_layers(session_id, tokens.len(), table, image)?;
        self.finish_prefill(tokens.len(), sampling)
    }

    fn execute_vision_layers(
        &mut self,
        session_id: Uuid,
        tokens: usize,
        table: &BlockTable,
        image: Option<ImageAttentionSpan>,
    ) -> Result<()> {
        for (index, layer) in self.layers.iter_mut().enumerate() {
            let plan = KvWritePlan::prefill(session_id, layer.layer(), table, 0, tokens)?;
            let (input, output) = if index.is_multiple_of(2) {
                (&self.prefill_first, &mut self.prefill_second)
            } else {
                (&self.prefill_second, &mut self.prefill_first)
            };
            layer.prepare_prefill(tokens)?;
            if let Some(graph) = self.decode_graph.as_mut() {
                graph.execute_prefill_masked(
                    index,
                    layer.prefill_plan(tokens)?,
                    input,
                    &plan,
                    table,
                    0,
                    output,
                    image,
                )?;
            } else {
                layer.execute_prefill_masked(input, output, &plan, table, 0, tokens, image)?;
            }
        }
        Ok(())
    }
}
