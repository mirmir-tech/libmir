use super::HybridLinearMoeModel;
use crate::engine::{Array, DecoderCache, Error, ImageTokenSpan, Result, Stream};

impl HybridLinearMoeModel {
    pub fn forward_multimodal_prefill(
        &self,
        token_ids: &Array,
        image_embeddings: &Array,
        image: ImageTokenSpan,
        position_ids: &Array,
        cache: &mut DecoderCache,
        stream: &Stream,
    ) -> Result<Array> {
        if cache.cached_tokens()? != 0 {
            return Err(Error::InvalidModel(
                "multimodal prefill requires an empty decoder cache".into(),
            ));
        }
        let token_shape = token_ids.shape()?;
        if token_shape.len() != 2 || token_shape[0] != 1 {
            return Err(Error::InvalidModel(format!(
                "hybrid linear-MoE multimodal token IDs must have shape [1, sequence], got {token_shape:?}"
            )));
        }
        let sequence = usize::try_from(token_shape[1])?;
        let image = ImageTokenSpan::new(image.start..image.end, sequence)?;
        let expected_image = [1, i32::try_from(image.len())?, i32::try_from(self.hidden_size)?];
        if image_embeddings.shape()? != expected_image {
            return Err(Error::InvalidModel(format!(
                "spatial-merge image embeddings must have shape {expected_image:?}"
            )));
        }
        if position_ids.shape()? != [3, i32::try_from(sequence)?] {
            return Err(Error::InvalidModel(
                "hybrid linear-MoE multimodal position IDs must have shape [3, sequence]".into(),
            ));
        }
        let hidden = self.embedding.lookup(token_ids, stream)?.slice_update(
            image_embeddings,
            &[0, image.start, 0],
            &[1, image.end, self.hidden_size],
            stream,
        )?;
        self.forward_embedded(hidden, cache, 0, true, Some(position_ids), stream)
    }
}
