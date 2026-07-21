use std::time::Instant;

use super::HybridMoeModel;
use crate::engine::{
    Array, DecoderCache, Error, ImageTokenSpan, Result, Stream,
    decoder::{LayerContext, LayerLoopOptions, forward_layers},
};

impl HybridMoeModel {
    pub fn forward_prefill(
        &self,
        token_ids: &Array,
        cache: &mut DecoderCache,
        position: i32,
        stream: &Stream,
    ) -> Result<Array> {
        self.forward_hidden(token_ids, cache, position, true, stream)
    }

    /// Runs an initial multimodal prefill without copying image embeddings
    /// back to the host.
    pub fn forward_multimodal_prefill(
        &self,
        token_ids: &Array,
        image_embeddings: &Array,
        image: ImageTokenSpan,
        bidirectional_image_attention: bool,
        cache: &mut DecoderCache,
        stream: &Stream,
    ) -> Result<Array> {
        if cache.cached_tokens()? != 0 {
            return Err(Error::InvalidModel(
                "multimodal prefill requires an empty decoder cache".into(),
            ));
        }
        let hidden = self.mixed_embeddings(token_ids, image_embeddings, image, stream)?;
        let image = bidirectional_image_attention.then_some(image);
        self.forward_embedded(hidden, cache, 0, true, image, stream)
    }

    pub fn mixed_embeddings(
        &self,
        token_ids: &Array,
        image_embeddings: &Array,
        image: ImageTokenSpan,
        stream: &Stream,
    ) -> Result<Array> {
        let token_shape = token_ids.shape()?;
        if token_shape.len() != 2 || token_shape[0] != 1 {
            return Err(Error::InvalidModel(format!(
                "multimodal token IDs must have shape [1, sequence], got {token_shape:?}"
            )));
        }
        let sequence = usize::try_from(token_shape[1])?;
        let image = ImageTokenSpan::new(image.start..image.end, sequence)?;
        let image_shape = image_embeddings.shape()?;
        let expected = [1, i32::try_from(image.len())?, i32::try_from(self.hidden_size)?];
        if image_shape != expected {
            return Err(Error::InvalidModel(format!(
                "image embeddings {image_shape:?} do not match expected {expected:?}"
            )));
        }
        let hidden = self
            .embedding
            .lookup(token_ids, stream)?
            .multiply_scalar(self.embed_scale, stream)?;
        hidden.slice_update(
            image_embeddings,
            &[0, image.start, 0],
            &[1, image.end, self.hidden_size],
            stream,
        )
    }

    pub(super) fn forward_hidden(
        &self,
        token_ids: &Array,
        cache: &mut DecoderCache,
        position: i32,
        causal: bool,
        stream: &Stream,
    ) -> Result<Array> {
        let profile = !causal && super::profile_components(stream);
        let started = Instant::now();
        let hidden = self
            .embedding
            .lookup(token_ids, stream)?
            .multiply_scalar(self.embed_scale, stream)?;
        if profile {
            hidden.async_eval()?;
            stream.synchronize()?;
            tracing::debug!(
                component = "embedding",
                milliseconds = started.elapsed().as_secs_f64() * 1_000.0,
                "MLX hybrid MoE component profile"
            );
        }
        self.forward_embedded(hidden, cache, position, causal, None, stream)
    }

    fn forward_embedded(
        &self,
        hidden: Array,
        cache: &mut DecoderCache,
        position: i32,
        causal: bool,
        image: Option<ImageTokenSpan>,
        stream: &Stream,
    ) -> Result<Array> {
        let profile = stream.config().diagnostics.profile_layers;
        let evaluation_step = causal
            .then_some(stream.config().diagnostics.prefill_evaluation_layers)
            .flatten();
        forward_layers(
            &self.layers,
            hidden,
            cache,
            LayerContext {
                position,
                causal,
                positions: None,
                image,
                stream,
            },
            LayerLoopOptions::new(profile, evaluation_step, false),
        )
    }
}
