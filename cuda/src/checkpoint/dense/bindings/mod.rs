use models::{
    semantic::{NormPlacement, SemanticModelSpec},
    weights::{
        DenseDecoderLayerBindings, DenseLayerFeedForwardBindings, DenseSandwichBindings,
        ParallelDecoderLayerBindings, PerLayerInputBindings, PostNormDecoderLayerBindings,
        SharedKeyValueLayerBindings, TensorBinding, WeightBindingPlan,
    },
};

use crate::{Error, Result};

mod attention;

pub use attention::DenseAttentionBindings;

/// A dense layer's bindings as the CUDA loader consumes them. Pre-norm
/// layers bind both input norms; post-norm layers (`OLMo` 2) bind neither and
/// normalize each sub-block's output through the sandwich norms instead;
/// parallel layers (Cohere) bind their one norm as both, which the
/// feed-forward applies to the layer input
/// ([`crate::FeedForwardInput::LayerInput`]). Gemma 4's E models add each
/// layer's per-layer input gate, and their later layers read an earlier
/// layer's keys and values ([`DenseAttentionBindings::Shared`]).
#[derive(Debug, Clone, Copy)]
pub struct DenseLayerBindings<'a> {
    pub input_norm: Option<&'a TensorBinding>,
    pub attention: DenseAttentionBindings<'a>,
    pub post_attention_norm: Option<&'a TensorBinding>,
    pub sandwich: Option<DenseSandwichBindings<'a>>,
    pub feed_forward: DenseLayerFeedForwardBindings<'a>,
    pub per_layer_input: Option<PerLayerInputBindings<'a>>,
}

impl<'a> DenseLayerBindings<'a> {
    /// The bindings of `layer` for the norm placement the semantic spec gives
    /// it.
    pub fn of_layer(
        plan: &'a WeightBindingPlan,
        semantic: &SemanticModelSpec,
        layer: usize,
    ) -> Result<Self> {
        let spec = semantic
            .decoder
            .layers
            .get(layer)
            .ok_or(Error::InvalidDecoderKernel("dense layer has no semantic spec"))?;
        let shared = spec.mixer.softmax_attention().and_then(|spec| spec.key_value_source);
        let bindings: Self = match (spec.norm_placement, shared) {
            (NormPlacement::Pre, Some(_)) => {
                plan.dense_decoder_layer_sharing_key_values(layer)?.into()
            },
            (_, Some(_)) => {
                return Err(Error::UnsupportedDecoderLayer(
                    "CUDA shares keys and values between pre-norm layers only".into(),
                ));
            },
            (NormPlacement::Pre, None) => plan.dense_decoder_layer(layer)?.into(),
            (NormPlacement::Post, None) => plan.post_norm_decoder_layer(layer)?.into(),
            (NormPlacement::Parallel, None) => plan.parallel_decoder_layer(layer)?.into(),
        };
        let per_layer_input = plan.per_layer_input(layer)?;
        if per_layer_input.is_some() != semantic.decoder.per_layer_input.is_some() {
            return Err(Error::UnsupportedDecoderLayer(format!(
                "layer {layer}'s per-layer input tensors differ from the decoder"
            )));
        }
        Ok(Self { per_layer_input, ..bindings })
    }

    /// Every binding of the layer.
    pub fn bindings(self) -> Vec<&'a TensorBinding> {
        let mut bindings = self.attention.projections();
        bindings.extend(self.feed_forward.bindings());
        if let Some(input) = self.per_layer_input {
            bindings.extend([input.gate, input.projection]);
        }
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
        norms.extend(self.attention.query_norm().into_iter().chain(self.attention.key_norm()));
        if let Some(sandwich) = self.sandwich {
            norms.extend([sandwich.attention_output_norm, sandwich.feed_forward_output_norm]);
        }
        norms.extend(self.per_layer_input.map(|input| input.output_norm));
        // A parallel layer binds its one norm twice.
        norms.dedup_by(|first, second| std::ptr::eq(*first, *second));
        norms
    }
}

impl<'a> From<DenseDecoderLayerBindings<'a>> for DenseLayerBindings<'a> {
    fn from(bindings: DenseDecoderLayerBindings<'a>) -> Self {
        Self {
            input_norm: Some(bindings.input_norm),
            attention: DenseAttentionBindings::Projected(bindings.attention),
            post_attention_norm: Some(bindings.post_attention_norm),
            sandwich: bindings.sandwich,
            feed_forward: bindings.feed_forward,
            per_layer_input: None,
        }
    }
}

impl<'a> From<PostNormDecoderLayerBindings<'a>> for DenseLayerBindings<'a> {
    fn from(bindings: PostNormDecoderLayerBindings<'a>) -> Self {
        Self {
            input_norm: None,
            attention: DenseAttentionBindings::Projected(bindings.attention),
            post_attention_norm: None,
            sandwich: Some(DenseSandwichBindings {
                attention_output_norm: bindings.attention_output_norm,
                feed_forward_output_norm: bindings.feed_forward_output_norm,
                layer_scale: None,
            }),
            feed_forward: bindings.feed_forward,
            per_layer_input: None,
        }
    }
}

impl<'a> From<ParallelDecoderLayerBindings<'a>> for DenseLayerBindings<'a> {
    fn from(bindings: ParallelDecoderLayerBindings<'a>) -> Self {
        Self {
            input_norm: Some(bindings.input_norm),
            attention: DenseAttentionBindings::Projected(bindings.attention),
            post_attention_norm: Some(bindings.input_norm),
            sandwich: None,
            feed_forward: bindings.feed_forward,
            per_layer_input: None,
        }
    }
}

impl<'a> From<SharedKeyValueLayerBindings<'a>> for DenseLayerBindings<'a> {
    fn from(bindings: SharedKeyValueLayerBindings<'a>) -> Self {
        Self {
            input_norm: Some(bindings.input_norm),
            attention: DenseAttentionBindings::shared(&bindings),
            post_attention_norm: Some(bindings.post_attention_norm),
            sandwich: bindings.sandwich,
            feed_forward: bindings.feed_forward,
            per_layer_input: None,
        }
    }
}
