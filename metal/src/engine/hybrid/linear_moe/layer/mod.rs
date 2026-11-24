mod mixer;
mod moe;
mod profile;

use std::time::Instant;

use models::{
    layout::DecoderConfig,
    semantic::DecoderLayerSpec,
    weights::{LatentDecoderLayerBindings, MixedDecoderLayerBindings},
};

pub(super) use self::mixer::Attention;
use self::profile::emit_component;
use super::{
    super::super::{
        Array, DecoderCache, ExpertFusion, ModelTensors, NormWeight, Result, Stream,
        binding::adjusted_norm,
    },
    feed_forward::FeedForward,
};
use crate::engine::{
    decoder::{LayerContext, LoweredLayer, MixerKind},
    lowering::LayerLowering,
};

impl LoweredLayer for HybridLinearMoeLayer {
    fn mixer_kind(&self) -> MixerKind {
        self.attention.mixer_kind()
    }

    fn forward_mixer(
        &self,
        input: &Array,
        cache: &mut DecoderCache,
        _index: usize,
        context: LayerContext<'_>,
    ) -> Result<Array> {
        self.mix(
            input,
            cache,
            context.position,
            context.causal,
            context.positions,
            context.stream,
        )
    }

    fn forward_feed_forward(&self, input: &Array, context: LayerContext<'_>) -> Result<Array> {
        self.feed_forward(input, context.stream)
    }
}

#[derive(Debug)]
pub(super) struct HybridLinearMoeLayer {
    pub(super) index: usize,
    pub(super) attention: Attention,
    pub(super) input_norm: NormWeight,
    pub(super) post_attention_norm: NormWeight,
    pub(super) moe: FeedForward,
    pub(super) rms_norm_eps: f32,
}

impl HybridLinearMoeLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn load(
        tensors: &ModelTensors,
        decoder: &DecoderConfig,
        layer: &DecoderLayerSpec,
        index: usize,
        bindings: MixedDecoderLayerBindings<'_>,
        lowering: LayerLowering,
        norm_shift: f32,
        stream: &Stream,
    ) -> Result<Self> {
        let attention =
            Attention::load(tensors, decoder, layer, bindings.mixer, norm_shift, stream)?;
        let moe = moe::load(tensors, layer, bindings.feed_forward, lowering, stream)?;
        Ok(Self {
            index,
            attention,
            input_norm: adjusted_norm(tensors, bindings.input_norm, norm_shift, stream)?,
            post_attention_norm: adjusted_norm(
                tensors,
                bindings.post_attention_norm,
                norm_shift,
                stream,
            )?,
            moe,
            rms_norm_eps: layer.input_norm.epsilon.to_string().parse()?,
        })
    }

    /// A latent attention layer; its norms carry no unit offset.
    pub(super) fn load_latent(
        tensors: &ModelTensors,
        layer: &DecoderLayerSpec,
        index: usize,
        bindings: LatentDecoderLayerBindings<'_>,
        lowering: LayerLowering,
        stream: &Stream,
    ) -> Result<Self> {
        Ok(Self {
            index,
            attention: Attention::load_latent(tensors, layer, bindings.attention, stream)?,
            input_norm: NormWeight::load_name(tensors, &bindings.input_norm.source)?,
            post_attention_norm: NormWeight::load_name(
                tensors,
                &bindings.post_attention_norm.source,
            )?,
            moe: moe::load(tensors, layer, bindings.feed_forward, lowering, stream)?,
            rms_norm_eps: layer.input_norm.epsilon.to_string().parse()?,
        })
    }

    fn mix(
        &self,
        input: &Array,
        cache: &mut DecoderCache,
        position: i32,
        causal: bool,
        positions: Option<&Array>,
        stream: &Stream,
    ) -> Result<Array> {
        let profile = stream.config().diagnostics.profile_components;
        let attention_started = Instant::now();
        let normalized = self.input_norm.apply(input, self.rms_norm_eps, stream)?;
        let attention = self.attention.forward(
            &normalized,
            cache,
            self.index,
            LayerContext {
                position,
                causal,
                positions,
                image: None,
                stream,
            },
        )?;
        if profile {
            emit_component(
                &attention,
                stream,
                self.index,
                self.attention_kind(),
                "attention",
                attention_started,
            )?;
        }
        input.add(&attention, stream)
    }

    fn feed_forward(&self, input: &Array, stream: &Stream) -> Result<Array> {
        let profile = stream.config().diagnostics.profile_components;
        let moe_started = Instant::now();
        let normalized = self.post_attention_norm.apply(input, self.rms_norm_eps, stream)?;
        let moe = self.moe.forward(&normalized, stream)?;
        if profile {
            emit_component(&moe, stream, self.index, self.attention_kind(), "moe", moe_started)?;
        }
        input.add(&moe, stream)
    }

    pub(super) const fn attention_kind(&self) -> &'static str {
        self.attention.kind()
    }

    pub(super) fn has_fused_expert_gate_up(&self) -> bool {
        self.moe.has_fused_routed_gate_up()
    }

    pub(super) fn has_decode_plan_candidates(&self) -> bool {
        self.moe.has_decode_plan_candidates()
    }

    pub(super) fn decode_plan_candidate_bytes(&self) -> Result<Option<usize>> {
        self.moe.decode_plan_candidate_bytes()
    }

    pub(super) fn enable_decode_plan_candidate(&mut self, stream: &Stream) -> Result<bool> {
        self.moe.enable_decode_plan_candidate(stream)
    }
}

impl ExpertFusion for HybridLinearMoeLayer {
    fn enable_expert_fusion(&mut self, stream: &Stream) -> Result<bool> {
        self.moe.enable_routed_gate_up(stream)
    }

    fn expert_fusion_bytes(&self) -> Result<Option<usize>> {
        self.moe.fused_routed_gate_up_bytes()
    }
}
