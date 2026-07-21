use std::time::Instant;

use models::{
    layout::DecoderConfig,
    weights::{HybridDecoderLayerBindings, HybridMixerBindings},
};

use super::super::{
    Array, DecoderCache, Error, ExpertFusion, GatedDeltaLayer, GatedDeltaLayerConfig,
    GatedFullAttention, GatedFullAttentionConfig, ModelTensors, NormWeight, Result,
    SharedExpertMoe, SharedExpertMoeConfig, Stream, binding::adjusted_norm,
    paged_attention_min_context,
};
use crate::engine::{
    decoder::{LayerContext, LoweredLayer, MixerKind},
    lowering::LayerLowering,
};

#[derive(Debug)]
pub(super) enum Attention {
    Linear(GatedDeltaLayer),
    Full(GatedFullAttention),
}

impl LoweredLayer for HybridLinearMoeLayer {
    fn mixer_kind(&self) -> MixerKind {
        match self.attention {
            Attention::Linear(_) => MixerKind::Linear,
            Attention::Full(_) => MixerKind::Softmax,
        }
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
    pub(super) moe: SharedExpertMoe,
    pub(super) rms_norm_eps: f32,
}

impl HybridLinearMoeLayer {
    pub(super) fn load(
        tensors: &ModelTensors,
        decoder: &DecoderConfig,
        index: usize,
        bindings: HybridDecoderLayerBindings<'_>,
        lowering: LayerLowering,
        norm_shift: f32,
        stream: &Stream,
    ) -> Result<Self> {
        let attention = attention(tensors, decoder, bindings.mixer, norm_shift, stream)?;
        let experts = SharedExpertMoeConfig::new(
            decoder
                .num_experts
                .ok_or_else(|| Error::InvalidModel("missing MoE expert count".into()))?,
            decoder
                .top_k_experts
                .ok_or_else(|| Error::InvalidModel("missing MoE top-k".into()))?,
        )?;
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
            moe: SharedExpertMoe::load_bindings(
                tensors,
                bindings.feed_forward,
                experts,
                lowering.feed_forward,
                stream,
            )?,
            rms_norm_eps: decoder.rms_norm_eps.to_string().parse()?,
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
        let attention = match &self.attention {
            Attention::Linear(layer) => {
                layer.forward(&normalized, cache.gated_delta_state(self.index)?, stream)?
            },
            Attention::Full(layer) => layer.forward_with_positions(
                &normalized,
                cache.full_attention_cache(self.index)?,
                paged_attention_min_context(stream),
                position,
                causal,
                positions,
                stream,
            )?,
        };
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
        match self.attention {
            Attention::Linear(_) => "gated_delta",
            Attention::Full(_) => "gated_full",
        }
    }

    pub(super) const fn has_fused_expert_gate_up(&self) -> bool {
        self.moe.has_fused_routed_gate_up()
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

fn emit_component(
    output: &Array,
    stream: &Stream,
    layer: usize,
    attention: &str,
    component: &str,
    started: Instant,
) -> Result<()> {
    output.async_eval()?;
    stream.synchronize()?;
    tracing::debug!(
        layer,
        attention,
        component,
        milliseconds = started.elapsed().as_secs_f64() * 1_000.0,
        "MLX hybrid linear MoE component profile"
    );
    Ok(())
}

fn attention(
    tensors: &ModelTensors,
    decoder: &DecoderConfig,
    bindings: HybridMixerBindings<'_>,
    norm_shift: f32,
    stream: &Stream,
) -> Result<Attention> {
    match bindings {
        HybridMixerBindings::Linear(bindings) => {
            let config = GatedDeltaLayerConfig::from_linear_attention(
                decoder.linear_attention.as_ref().ok_or_else(|| {
                    Error::InvalidModel("missing linear attention configuration".into())
                })?,
                decoder.rms_norm_eps,
            )?;
            GatedDeltaLayer::load_bindings(tensors, bindings, config, norm_shift, stream)
                .map(Attention::Linear)
        },
        HybridMixerBindings::Softmax(bindings) => GatedFullAttention::load_bindings(
            tensors,
            bindings,
            GatedFullAttentionConfig::from_decoder(decoder)?,
            norm_shift,
            stream,
        )
        .map(Attention::Full),
    }
}
