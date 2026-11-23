//! Decoder blocks without a norm between their sub-blocks: post-norm
//! (`OLMo` 2: `x + norm(attention(x))`, then `h + norm(feed_forward(h))`) and
//! parallel (Cohere: `x + attention(norm(x)) + feed_forward(norm(x))`).

use super::{
    DenseLayerFeedForwardBindings, DenseSoftmaxBindings, LayerTensorRole, TensorBinding,
    WeightBindingPlan, feed_forward, layer, softmax,
};
use crate::error::Result;

#[derive(Debug, Clone, Copy)]
pub struct PostNormDecoderLayerBindings<'a> {
    pub attention: DenseSoftmaxBindings<'a>,
    /// Normalizes the attention output before its residual add.
    pub attention_output_norm: &'a TensorBinding,
    pub feed_forward: DenseLayerFeedForwardBindings<'a>,
    /// Normalizes the feed-forward output before its residual add.
    pub feed_forward_output_norm: &'a TensorBinding,
}

#[derive(Debug, Clone, Copy)]
pub struct ParallelDecoderLayerBindings<'a> {
    /// Normalizes the stream both sub-blocks read.
    pub input_norm: &'a TensorBinding,
    pub attention: DenseSoftmaxBindings<'a>,
    pub feed_forward: DenseLayerFeedForwardBindings<'a>,
}

impl WeightBindingPlan {
    pub fn parallel_decoder_layer(&self, index: usize) -> Result<ParallelDecoderLayerBindings<'_>> {
        Ok(ParallelDecoderLayerBindings {
            input_norm: layer(self, index, LayerTensorRole::InputNorm)?,
            attention: softmax(self, index)?,
            feed_forward: feed_forward(self, index)?,
        })
    }

    pub fn post_norm_decoder_layer(
        &self,
        index: usize,
    ) -> Result<PostNormDecoderLayerBindings<'_>> {
        Ok(PostNormDecoderLayerBindings {
            attention: softmax(self, index)?,
            attention_output_norm: layer(self, index, LayerTensorRole::PostAttentionNorm)?,
            feed_forward: feed_forward(self, index)?,
            feed_forward_output_norm: layer(self, index, LayerTensorRole::PostFeedForwardNorm)?,
        })
    }
}
