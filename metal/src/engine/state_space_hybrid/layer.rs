use models::{
    layout::DecoderConfig,
    weights::{StateSpaceHybridLayerBindings, StateSpaceHybridMixerBindings},
};

use super::{attention::PlainAttention, feed_forward::FeedForward, mamba::Mamba2Mixer};
use crate::engine::{
    Array, DecoderCache, Error, ModelTensors, NormWeight, Result, Stream,
    decoder::{LayerContext, LoweredLayer, MixerKind},
};

#[derive(Debug)]
pub(super) enum Mixer {
    Mamba(Box<Mamba2Mixer>),
    Attention(Box<PlainAttention>),
}

#[derive(Debug)]
pub(super) struct StateSpaceHybridLayer {
    pub(super) index: usize,
    pub(super) input_norm: NormWeight,
    pub(super) mixer: Mixer,
    pub(super) feed_forward: Option<FeedForward>,
    pub(super) epsilon: f32,
}

impl StateSpaceHybridLayer {
    pub(super) fn load(
        tensors: &ModelTensors,
        decoder: &DecoderConfig,
        index: usize,
        bindings: StateSpaceHybridLayerBindings<'_>,
        epsilon: f32,
        stream: &Stream,
    ) -> Result<Self> {
        let hybrid = decoder.state_space.as_ref().ok_or_else(|| {
            Error::InvalidModel("missing state-space hybrid configuration".into())
        })?;
        let mixer = match bindings.mixer {
            StateSpaceHybridMixerBindings::StateSpace(mixer) => Mixer::Mamba(Box::new(
                Mamba2Mixer::load(tensors, mixer, &hybrid.mamba, epsilon, stream)?,
            )),
            StateSpaceHybridMixerBindings::Attention(attention) => Mixer::Attention(Box::new(
                PlainAttention::load(tensors, attention, decoder, stream)?,
            )),
        };
        let feed_forward = bindings
            .feed_forward
            .map(|bindings| FeedForward::load(tensors, bindings, hybrid, epsilon, stream))
            .transpose()?;
        Ok(Self {
            index,
            input_norm: NormWeight::load_name(tensors, &bindings.input_norm.source)?,
            mixer,
            feed_forward,
            epsilon,
        })
    }

    pub(super) fn feed_forward(&self, input: &Array, stream: &Stream) -> Result<Array> {
        let Some(feed_forward) = &self.feed_forward else {
            return Array::from_native(input.native().clone());
        };
        let (norm, epsilon) = feed_forward.norm();
        let normalized = norm.apply(input, epsilon, stream)?;
        input.add(&feed_forward.forward(&normalized, stream)?, stream)
    }

    fn mix(
        &self,
        input: &Array,
        cache: &mut DecoderCache,
        causal: bool,
        stream: &Stream,
    ) -> Result<Array> {
        let normalized = self.input_norm.apply(input, self.epsilon, stream)?;
        let mixed = match &self.mixer {
            Mixer::Mamba(mixer) => {
                let mut states = [cache.state_space_state(self.index)?];
                mixer.forward(&normalized, &mut states, stream)?
            },
            Mixer::Attention(attention) => {
                let cache = cache.full_attention_cache(self.index)?;
                attention.forward(&normalized, cache, causal, stream)?
            },
        };
        input.add(&mixed, stream)
    }

    pub(super) fn mix_packed(
        &self,
        input: &Array,
        caches: &mut [&mut DecoderCache],
        causal: bool,
        stream: &Stream,
    ) -> Result<Array> {
        let normalized = self.input_norm.apply(input, self.epsilon, stream)?;
        let mixed = match &self.mixer {
            Mixer::Mamba(mixer) => {
                let mut states = caches
                    .iter_mut()
                    .map(|cache| cache.state_space_state(self.index))
                    .collect::<Result<Vec<_>>>()?;
                mixer.forward(&normalized, &mut states, stream)?
            },
            Mixer::Attention(attention) => {
                attention.forward_packed(&normalized, caches, self.index, causal, stream)?
            },
        };
        input.add(&mixed, stream)
    }
}

impl LoweredLayer for StateSpaceHybridLayer {
    fn mixer_kind(&self) -> MixerKind {
        match self.mixer {
            Mixer::Mamba(_) => MixerKind::Linear,
            Mixer::Attention(_) => MixerKind::Softmax,
        }
    }

    fn forward_mixer(
        &self,
        input: &Array,
        cache: &mut DecoderCache,
        _index: usize,
        context: LayerContext<'_>,
    ) -> Result<Array> {
        self.mix(input, cache, context.causal, context.stream)
    }

    fn forward_feed_forward(&self, input: &Array, context: LayerContext<'_>) -> Result<Array> {
        self.feed_forward(input, context.stream)
    }
}
