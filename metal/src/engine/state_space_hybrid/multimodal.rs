//! Prefill of a prompt whose image tokens take a vision encoder's soft tokens.
//! Attention here carries no positional encoding and the Mamba-2 mixers read
//! only the sequence order, so the image enters with plain positions.

use super::StateSpaceHybridModel;
use crate::engine::{Array, DecoderCache, Error, ImageTokenSpan, Result, Stream};

impl StateSpaceHybridModel {
    /// Runs an initial causal prefill with `image_embeddings` replacing the
    /// embeddings of the image span, without copying them back to the host.
    pub fn forward_multimodal_prefill(
        &self,
        token_ids: &Array,
        image_embeddings: &Array,
        image: ImageTokenSpan,
        cache: &mut DecoderCache,
        stream: &Stream,
    ) -> Result<Array> {
        if cache.cached_tokens()? != 0 {
            return Err(Error::InvalidModel(
                "multimodal prefill requires an empty decoder cache".into(),
            ));
        }
        let token_shape = token_ids.shape()?;
        let [1, sequence] = token_shape[..] else {
            return Err(Error::InvalidModel(format!(
                "multimodal token IDs must have shape [1, sequence], got {token_shape:?}"
            )));
        };
        let image = ImageTokenSpan::new(image.start..image.end, usize::try_from(sequence)?)?;
        let hidden = self.embedding.lookup(token_ids, stream)?;
        let hidden_size = *hidden.shape()?.get(2).ok_or_else(|| {
            Error::InvalidModel("state-space hybrid embeddings have no hidden axis".into())
        })?;
        let image_shape = image_embeddings.shape()?;
        let expected = [1, i32::try_from(image.len())?, hidden_size];
        if image_shape != expected {
            return Err(Error::InvalidModel(format!(
                "image embeddings {image_shape:?} do not match expected {expected:?}"
            )));
        }
        let hidden = hidden.slice_update(
            &image_embeddings.astype_like(&hidden, stream)?,
            &[0, image.start, 0],
            &[1, image.end, usize::try_from(hidden_size)?],
            stream,
        )?;
        self.forward_embedded(hidden, cache, 0, true, stream)
    }
}
