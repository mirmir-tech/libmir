use models::vision::{PixelShufflePreprocessedImage, PixelShufflePromptTokens};
use runtime::backend::SamplingLogits;
use uuid::Uuid;

use super::super::{NativePrefill, evaluation};
use crate::{
    MetalProgressEvent,
    native::{
        error::{Error, Result},
        model::{LoadedModel, LoadedVisionModel},
        session::SessionState,
        step,
    },
};

impl LoadedModel {
    /// The prompt up to its last token through the pixel-shuffle tower and the
    /// decoder in one causal pass, then the last token as a decode step. Plain
    /// positions: the image layout never enters the decoder's positions.
    pub(in crate::native) fn prefill_pixel_shuffle_vision(
        &mut self,
        session: Uuid,
        prompt: &PixelShufflePromptTokens,
        image: &PixelShufflePreprocessedImage,
        blocks: &[runtime::kv::BlockId],
        sampling: &SamplingLogits,
        progress: &mut dyn FnMut(MetalProgressEvent),
    ) -> Result<NativePrefill> {
        let Some((&last, prefix)) = prompt.token_ids.split_last() else {
            return Err(Error::EmptyPrompt);
        };
        if prompt.image_end > prefix.len() {
            return Err(Error::UnsupportedModel(
                "pixel-shuffle vision image block must precede the final prompt token".into(),
            ));
        }
        let length = prompt.token_ids.len();
        let mut state = SessionState::new(self.execution.decoder()?.new_cache(&self.stream)?);
        if !blocks.is_empty() {
            state.cache.bind_blocks(blocks, None, &self.stream)?;
        }
        let model = self.execution.decoder()?;
        let Some(LoadedVisionModel::PixelShuffle(tower)) = self.vision_model.as_ref() else {
            return Err(Error::UnsupportedModel(
                "pixel-shuffle vision tower is not loaded or its tensors are incomplete".into(),
            ));
        };
        progress(MetalProgressEvent::prefill_tokens(0, length));
        let prefix_prompt = PixelShufflePromptTokens {
            token_ids: prefix.to_vec(),
            image_start: prompt.image_start,
            image_end: prompt.image_end,
        };
        let hidden = tower.forward_multimodal_prefill(
            model, &prefix_prompt, image, &mut state.cache, &self.stream,
        )?;
        let mut roots = vec![&hidden];
        state.cache.extend_graph_roots(&mut roots);
        self.stream.eval_many(&roots)?;
        self.settle_prefill_graph()?;
        state.cache.detach_evaluated_graphs(&self.stream)?;
        progress(MetalProgressEvent::prefill_tokens(prefix.len(), length));
        let greedy = *sampling == SamplingLogits::None;
        let logits =
            step::forward_token(model, &self.stream, &mut state, last, prefix.len(), greedy)?;
        evaluation::materialize(self, &state, &logits)?;
        state.position = length;
        let output = step::output(model, &self.stream, &mut state, logits, sampling)?;
        self.sessions.insert(session, state);
        progress(MetalProgressEvent::prefill_tokens(length, length));
        Ok(NativePrefill { output, prefix_cache_tokens: 0 })
    }
}
