use std::time::Instant;

use super::{HybridMoeLayer, profile_components};
use crate::engine::{
    Array, KvCache, Result, Stream,
    hybrid_moe::{
        attention::{self, DecodeContext},
        feed_forward,
    },
};

impl HybridMoeLayer {
    pub fn forward_uncached_decode(&self, input: &Array, stream: &Stream) -> Result<Array> {
        self.forward_decode(input, None, 0, false, stream)
    }

    pub fn forward_decode(
        &self,
        input: &Array,
        cache: Option<&mut KvCache>,
        position: i32,
        causal: bool,
        stream: &Stream,
    ) -> Result<Array> {
        let hidden = self.mix(input, cache, position, causal, None, stream)?;
        self.feed_forward(&hidden, causal, stream)
    }

    pub(super) fn mix(
        &self,
        input: &Array,
        cache: Option<&mut KvCache>,
        position: i32,
        causal: bool,
        mask: Option<&Array>,
        stream: &Stream,
    ) -> Result<Array> {
        let profile = !causal && profile_components(stream);
        let started = Instant::now();
        let normalized = self.weights.input_norm.apply(input, self.config.rms_norm_eps, stream)?;
        let attention = attention::forward_decode(
            &normalized,
            &self.weights.attention,
            self.config,
            self.fused_attention.as_ref(),
            self.fused_key_value.as_ref(),
            DecodeContext { cache, position, causal, mask, stream },
        )?;
        let attention =
            self.weights
                .post_attention_norm
                .apply(&attention, self.config.rms_norm_eps, stream)?;
        let hidden = input.add(&attention, stream)?;
        if profile {
            emit_profile(&hidden, stream, self.config.layer_index, "attention", started)?;
        }
        Ok(hidden)
    }

    pub(super) fn feed_forward(
        &self,
        input: &Array,
        causal: bool,
        stream: &Stream,
    ) -> Result<Array> {
        let profile = !causal && profile_components(stream);
        let started = Instant::now();
        let feed_forward = feed_forward::forward(
            input,
            &self.weights,
            self.config,
            self.fused_gate_up.as_ref(),
            self.fused_expert_gate_up.as_ref(),
            stream,
        )?;
        let feed_forward = self.weights.post_feed_forward_norm.apply(
            &feed_forward,
            self.config.rms_norm_eps,
            stream,
        )?;
        let output =
            input.add(&feed_forward, stream)?.multiply(&self.weights.layer_scalar, stream)?;
        if profile {
            emit_profile(&output, stream, self.config.layer_index, "feed_forward", started)?;
        }
        Ok(output)
    }
}

pub(in crate::engine::hybrid_moe) fn emit_profile(
    output: &Array,
    stream: &Stream,
    layer: usize,
    component: &str,
    started: Instant,
) -> Result<()> {
    output.async_eval()?;
    stream.synchronize()?;
    tracing::debug!(layer, component, "MLX hybrid MoE component synchronized");
    tracing::debug!(
        layer,
        component,
        milliseconds = started.elapsed().as_secs_f64() * 1_000.0,
        "MLX hybrid MoE component profile"
    );
    Ok(())
}
