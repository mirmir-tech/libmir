#![allow(clippy::print_stdout)]

use std::{env, hint::black_box, path::PathBuf, time::Instant};

use models::layout::{DecoderConfig, ModelLayout};

use super::*;
use crate::engine::Error;

#[test]
#[ignore = "benchmark; set MIRMIR_BENCH_MODEL or MODEL"]
fn bench_hybrid_moe_attention_decode() -> Result<()> {
    let root = model_root()?;
    let layout = ModelLayout::inspect(&root)?;
    let decoder = DecoderConfig::from_layout(&layout)?;
    let load_stream = Stream::new_cpu()?;
    let tensors = ModelTensors::load(root, &load_stream)?;
    let stream = Stream::new_gpu()?;
    let config =
        HybridMoeLayerConfig::from_decoder(env_usize("MIRMIR_BENCH_LAYER", 0)?, &decoder, 64)?;
    let layer = HybridMoeLayer::load(&tensors, config, &stream)?;
    let reference = tensors.get("language_model.model.norm.weight")?;
    let prefill = benchmark_input(128, decoder.hidden_size, &reference, &stream)?;
    let mut cache = KvCache::new_with_window(256, config.max_context)?;
    let warmup = layer.attention_residual_for_test(&prefill, &mut cache, 0, true, &stream)?;
    warmup.async_eval()?;
    stream.synchronize()?;
    let decode = benchmark_input(1, decoder.hidden_size, &reference, &stream)?;
    let iters = env_usize("MIRMIR_BENCH_ITERS", 20)?;
    let warmup = env_usize("MIRMIR_BENCH_WARMUP", 5)?;
    for offset in 0..warmup {
        eval_attention(&layer, &decode, &mut cache, 128 + offset, &stream)?;
    }
    let started = Instant::now();
    for offset in 0..iters {
        eval_attention(&layer, &decode, &mut cache, 128 + warmup + offset, &stream)?;
    }
    let iters = f64::from(u32::try_from(iters)?);
    println!(
        "mirmir_attention_bench layer={} context=128 decode_ms={:.4}",
        config.layer_index,
        started.elapsed().as_secs_f64() * 1_000.0 / iters
    );
    Ok(())
}

impl HybridMoeLayer {
    pub(crate) fn attention_residual_for_test(
        &self,
        input: &Array,
        cache: &mut KvCache,
        position: i32,
        causal: bool,
        stream: &Stream,
    ) -> Result<Array> {
        let normalized = self.weights.input_norm.apply(input, self.config.rms_norm_eps, stream)?;
        let attention = attention::forward_decode(
            &normalized,
            &self.weights.attention,
            self.config,
            self.fused_attention.as_ref(),
            self.fused_key_value.as_ref(),
            DecodeContext {
                cache: Some(cache),
                position,
                causal,
                stream,
            },
        )?;
        let attention =
            self.weights
                .post_attention_norm
                .apply(&attention, self.config.rms_norm_eps, stream)?;
        input.add(&attention, stream)
    }

    pub(crate) fn routing_for_test(
        &self,
        input: &Array,
        stream: &Stream,
    ) -> Result<crate::engine::RouterOutput> {
        feed_forward::routing(input, &self.weights, self.config, stream)
    }

    pub(crate) fn router_scores_for_test(
        &self,
        input: &Array,
        stream: &Stream,
    ) -> Result<(Array, Array)> {
        let normalized =
            input.rms_norm(&self.weights.router.norm_scale, self.config.rms_norm_eps, stream)?;
        let scores = self.weights.router.projection.forward(&normalized, stream)?;
        Ok((normalized, scores))
    }

    pub(crate) fn query_for_test(
        &self,
        input: &Array,
        position: i32,
        stream: &Stream,
    ) -> Result<(Array, Array)> {
        let input = self.weights.input_norm.apply(input, self.config.rms_norm_eps, stream)?;
        let query = self.fused_attention.as_ref().map_or_else(
            || self.weights.attention.query.forward(&input, stream),
            |fused| Ok(fused.forward(&input, stream)?.query),
        )?;
        let query =
            query.reshape(&[1, 1, self.config.attention_heads, self.config.head_dim], stream)?;
        let query =
            self.weights
                .attention
                .query_norm
                .apply(&query, self.config.rms_norm_eps, stream)?;
        let rotated = attention::rope_layout(
            &query,
            self.weights.attention.rope_frequencies.as_ref(),
            self.config,
            position,
            stream,
        )?;
        Ok((query, rotated))
    }

    pub(crate) fn key_value_for_test(
        &self,
        input: &Array,
        position: i32,
        stream: &Stream,
    ) -> Result<(Array, Array)> {
        let input = self.weights.input_norm.apply(input, self.config.rms_norm_eps, stream)?;
        let raw_keys = self.weights.attention.key.forward(&input, stream)?;
        let raw_values = self
            .weights
            .attention
            .value
            .as_ref()
            .map(|value| value.forward(&input, stream))
            .transpose()?;
        let sequence =
            input.shape()?.get(1).copied().ok_or_else(|| {
                Error::InvalidModel("attention input has no sequence axis".into())
            })?;
        let keys =
            raw_keys.reshape(&[1, sequence, self.config.kv_heads, self.config.head_dim], stream)?;
        let keys =
            self.weights.attention.key_norm.apply(&keys, self.config.rms_norm_eps, stream)?;
        let keys = attention::rope_layout(
            &keys,
            self.weights.attention.rope_frequencies.as_ref(),
            self.config,
            position,
            stream,
        )?;
        let values = raw_values
            .ok_or_else(|| Error::InvalidModel("missing hybrid MoE value projection".into()))?
            .reshape(&[1, sequence, self.config.kv_heads, self.config.head_dim], stream)?
            .rms_norm_unit(self.config.rms_norm_eps, stream)?
            .transpose(&[0, 2, 1, 3], stream)?;
        Ok((keys, values))
    }

    pub(crate) fn rope_frequencies_for_test(&self) -> Option<&Array> {
        self.weights.attention.rope_frequencies.as_ref()
    }

    pub(crate) fn feed_forward_for_test(&self, input: &Array, stream: &Stream) -> Result<Array> {
        let output = feed_forward::forward(
            input,
            &self.weights,
            self.config,
            self.fused_gate_up.as_ref(),
            self.fused_expert_gate_up.as_ref(),
            stream,
        )?;
        self.weights
            .post_feed_forward_norm
            .apply(&output, self.config.rms_norm_eps, stream)
    }

    pub(crate) fn feed_forward_components_for_test(
        &self,
        input: &Array,
        stream: &Stream,
    ) -> Result<(Array, crate::engine::RouterOutput, Array, Array)> {
        let dense = feed_forward::dense(
            input,
            &self.weights,
            self.config,
            self.fused_gate_up.as_ref(),
            stream,
        )?;
        let routing = feed_forward::routing(input, &self.weights, self.config, stream)?;
        let experts = feed_forward::experts(
            input,
            &self.weights,
            self.config,
            self.fused_expert_gate_up.as_ref(),
            stream,
        )?;
        let output = dense.add(&experts, stream)?;
        let output =
            self.weights
                .post_feed_forward_norm
                .apply(&output, self.config.rms_norm_eps, stream)?;
        Ok((dense, routing, experts, output))
    }
}

fn eval_attention(
    layer: &HybridMoeLayer,
    input: &Array,
    cache: &mut KvCache,
    position: usize,
    stream: &Stream,
) -> Result<()> {
    let output =
        layer.attention_residual_for_test(input, cache, i32::try_from(position)?, false, stream)?;
    output.async_eval()?;
    stream.synchronize()?;
    black_box(output);
    Ok(())
}

fn benchmark_input(
    tokens: usize,
    hidden_size: usize,
    reference: &Array,
    stream: &Stream,
) -> Result<Array> {
    let length = tokens.checked_mul(hidden_size).ok_or(Error::ShapeOverflow)?;
    Array::from_f32(&vec![0.25; length], &[1, i32::try_from(tokens)?, i32::try_from(hidden_size)?])?
        .astype_like(reference, stream)
}

fn model_root() -> Result<PathBuf> {
    env::var_os("MIRMIR_BENCH_MODEL")
        .or_else(|| env::var_os("MODEL"))
        .map(PathBuf::from)
        .ok_or_else(|| Error::InvalidModel("set MIRMIR_BENCH_MODEL or MODEL".into()))
}

fn env_usize(name: &str, default: usize) -> Result<usize> {
    match env::var(name) {
        Ok(value) => Ok(value.parse()?),
        Err(_) => Ok(default),
    }
}
