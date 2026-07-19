use std::time::Instant;

use models::layout::{AttentionLayerType, DecoderConfig};

use super::super::{
    Array, DecoderCache, Error, ExpertFusion, GatedDeltaLayer, GatedDeltaLayerConfig,
    GatedFullAttention, GatedFullAttentionConfig, ModelTensors, NormWeight, Result,
    SharedExpertMoe, SharedExpertMoeConfig, Stream, paged_attention_min_context,
};

#[derive(Debug)]
pub(super) enum Attention {
    Linear(GatedDeltaLayer),
    Full(GatedFullAttention),
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
        group_size: i32,
        norm_shift: f32,
        stream: &Stream,
    ) -> Result<Self> {
        let prefix = format!("language_model.model.layers.{index}");
        let attention = attention(tensors, decoder, index, group_size, norm_shift, stream)?;
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
            input_norm: NormWeight::load_adjusted(
                tensors,
                &format!("{prefix}.input_layernorm"),
                norm_shift,
                stream,
            )?,
            post_attention_norm: NormWeight::load_adjusted(
                tensors,
                &format!("{prefix}.post_attention_layernorm"),
                norm_shift,
                stream,
            )?,
            moe: SharedExpertMoe::load(
                tensors,
                &format!("{prefix}.mlp"),
                experts,
                group_size,
                stream,
            )?,
            rms_norm_eps: decoder.rms_norm_eps.to_string().parse()?,
        })
    }

    pub(super) fn forward_with_positions(
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
        let hidden = input.add(&attention, stream)?;
        let moe_started = Instant::now();
        let normalized = self.post_attention_norm.apply(&hidden, self.rms_norm_eps, stream)?;
        let moe = self.moe.forward(&normalized, stream)?;
        if profile {
            emit_component(&moe, stream, self.index, self.attention_kind(), "moe", moe_started)?;
        }
        hidden.add(&moe, stream)
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
    index: usize,
    group_size: i32,
    norm_shift: f32,
    stream: &Stream,
) -> Result<Attention> {
    let prefix = format!("language_model.model.layers.{index}");
    match decoder.layer_type(index) {
        AttentionLayerType::Linear => {
            let config = GatedDeltaLayerConfig::from_linear_attention(
                decoder.linear_attention.as_ref().ok_or_else(|| {
                    Error::InvalidModel("missing linear attention configuration".into())
                })?,
                decoder.rms_norm_eps,
            )?;
            GatedDeltaLayer::load_with_norm_shift(
                tensors,
                &format!("{prefix}.linear_attn"),
                config,
                group_size,
                norm_shift,
                stream,
            )
            .map(Attention::Linear)
        },
        AttentionLayerType::Full => GatedFullAttention::load_with_norm_shift(
            tensors,
            &format!("{prefix}.self_attn"),
            GatedFullAttentionConfig::from_decoder(decoder)?,
            group_size,
            norm_shift,
            stream,
        )
        .map(Attention::Full),
        AttentionLayerType::Sliding => Err(Error::InvalidModel(
            "hybrid linear MoE does not support sliding attention layers".into(),
        )),
    }
}
