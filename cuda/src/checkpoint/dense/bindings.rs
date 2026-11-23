use models::{
    semantic::{NormPlacement, SemanticModelSpec},
    weights::{
        DenseDecoderLayerBindings, DenseLayerFeedForwardBindings, DenseSandwichBindings,
        DenseSoftmaxBindings, ParallelDecoderLayerBindings, PostNormDecoderLayerBindings,
        TensorBinding, WeightBindingPlan,
    },
};

use crate::{Error, Result};

/// A dense layer's bindings as the CUDA loader consumes them. Pre-norm
/// layers bind both input norms; post-norm layers (`OLMo` 2) bind neither and
/// normalize each sub-block's output through the sandwich norms instead;
/// parallel layers (Cohere) bind their one norm as both, which the
/// feed-forward applies to the layer input
/// ([`crate::FeedForwardInput::LayerInput`]).
#[derive(Debug, Clone, Copy)]
pub struct DenseLayerBindings<'a> {
    pub input_norm: Option<&'a TensorBinding>,
    pub attention: DenseSoftmaxBindings<'a>,
    pub post_attention_norm: Option<&'a TensorBinding>,
    pub sandwich: Option<DenseSandwichBindings<'a>>,
    pub feed_forward: DenseLayerFeedForwardBindings<'a>,
}

impl<'a> DenseLayerBindings<'a> {
    /// The bindings of `layer` for the norm placement the semantic spec gives
    /// it.
    pub fn of_layer(
        plan: &'a WeightBindingPlan,
        semantic: &SemanticModelSpec,
        layer: usize,
    ) -> Result<Self> {
        let placement = semantic
            .decoder
            .layers
            .get(layer)
            .ok_or(Error::InvalidDecoderKernel("dense layer has no semantic spec"))?
            .norm_placement;
        match placement {
            NormPlacement::Pre => Ok(plan.dense_decoder_layer(layer)?.into()),
            NormPlacement::Post => Ok(plan.post_norm_decoder_layer(layer)?.into()),
            NormPlacement::Parallel => Ok(plan.parallel_decoder_layer(layer)?.into()),
        }
    }

    /// Every binding of the layer.
    pub fn bindings(self) -> Vec<&'a TensorBinding> {
        let attention = self.attention;
        let mut bindings = vec![attention.query, attention.key, attention.value, attention.output];
        bindings.extend(self.feed_forward.bindings());
        bindings.extend(self.norms());
        bindings.extend(self.sandwich.and_then(|sandwich| sandwich.layer_scale));
        bindings
    }

    pub fn physical_sources(self) -> Vec<&'a str> {
        self.bindings().into_iter().flat_map(TensorBinding::physical_sources).collect()
    }

    /// The layer's norm tensors, in no particular order.
    pub fn norms(self) -> Vec<&'a TensorBinding> {
        let mut norms: Vec<_> =
            self.input_norm.into_iter().chain(self.post_attention_norm).collect();
        norms.extend(self.attention.query_norm.into_iter().chain(self.attention.key_norm));
        if let Some(sandwich) = self.sandwich {
            norms.extend([sandwich.attention_output_norm, sandwich.feed_forward_output_norm]);
        }
        // A parallel layer binds its one norm twice.
        norms.dedup_by(|first, second| std::ptr::eq(*first, *second));
        norms
    }
}

impl<'a> From<DenseDecoderLayerBindings<'a>> for DenseLayerBindings<'a> {
    fn from(bindings: DenseDecoderLayerBindings<'a>) -> Self {
        Self {
            input_norm: Some(bindings.input_norm),
            attention: bindings.attention,
            post_attention_norm: Some(bindings.post_attention_norm),
            sandwich: bindings.sandwich,
            feed_forward: bindings.feed_forward,
        }
    }
}

impl<'a> From<PostNormDecoderLayerBindings<'a>> for DenseLayerBindings<'a> {
    fn from(bindings: PostNormDecoderLayerBindings<'a>) -> Self {
        Self {
            input_norm: None,
            attention: bindings.attention,
            post_attention_norm: None,
            sandwich: Some(DenseSandwichBindings {
                attention_output_norm: bindings.attention_output_norm,
                feed_forward_output_norm: bindings.feed_forward_output_norm,
                layer_scale: None,
            }),
            feed_forward: bindings.feed_forward,
        }
    }
}

impl<'a> From<ParallelDecoderLayerBindings<'a>> for DenseLayerBindings<'a> {
    fn from(bindings: ParallelDecoderLayerBindings<'a>) -> Self {
        Self {
            input_norm: Some(bindings.input_norm),
            attention: bindings.attention,
            post_attention_norm: Some(bindings.input_norm),
            sandwich: None,
            feed_forward: bindings.feed_forward,
        }
    }
}
