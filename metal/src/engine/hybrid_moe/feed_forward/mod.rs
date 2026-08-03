use std::time::Instant;

use super::{
    HybridMoeLayerConfig,
    layer::{emit_profile, profile_components},
    weights::LayerWeights,
};
use crate::engine::{
    Array, FusedExpertGateUp, FusedGateUp, Result, Stream, fusion_planner::FusionPlanner,
    gate_up_tuning, lowering::FeedForwardLowering,
};

mod expert;

pub(super) use expert::experts;

pub(super) fn forward(
    input: &Array,
    weights: &LayerWeights,
    config: HybridMoeLayerConfig,
    fused_gate_up: Option<&FusedGateUp>,
    fused_expert_gate_up: Option<&FusedExpertGateUp>,
    stream: &Stream,
) -> Result<Array> {
    let profile = profile_components(stream) && input.shape()?[0] > 1;
    let started = Instant::now();
    let dense = dense(input, weights, config, fused_gate_up, stream)?;
    if profile {
        emit_profile(&dense, stream, config.layer_index, "dense", started)?;
    }
    let started = Instant::now();
    let experts = experts(input, weights, config, fused_expert_gate_up, stream)?;
    if profile {
        emit_profile(&experts, stream, config.layer_index, "experts", started)?;
    }
    dense.add(&experts, stream)
}

pub(super) fn dense(
    input: &Array,
    weights: &LayerWeights,
    config: HybridMoeLayerConfig,
    fused_gate_up: Option<&FusedGateUp>,
    stream: &Stream,
) -> Result<Array> {
    let hidden = weights.pre_dense_norm.apply(input, config.rms_norm_eps, stream)?;
    let fused = gate_up_tuning::is_single_token(input)?.then_some(fused_gate_up).flatten();
    let (gate, up) =
        gate_up_tuning::forward(&weights.dense.gate, &weights.dense.up, fused, &hidden, stream)?;
    let activated = gate.gelu_approx_mul(&up, stream)?;
    let output = weights.dense.down.forward(&activated, stream)?;
    weights.post_dense_norm.apply(&output, config.rms_norm_eps, stream)
}

pub(super) fn routing(
    input: &Array,
    weights: &LayerWeights,
    config: HybridMoeLayerConfig,
    stream: &Stream,
) -> Result<crate::engine::RouterOutput> {
    if FusionPlanner::new(stream).native_router(FeedForwardLowering::DenseAndRouted) {
        return weights.router.projection.route(
            input,
            &weights.router.norm_scale,
            &weights.router.expert_scale,
            config.rms_norm_eps,
            config.top_k,
            stream,
        );
    }
    let normalized = input.rms_norm(&weights.router.norm_scale, config.rms_norm_eps, stream)?;
    let scores = weights.router.projection.forward(&normalized, stream)?;
    scores.router_top_k(&weights.router.expert_scale, config.top_k, stream)
}

#[cfg(test)]
#[allow(clippy::print_stdout)]
mod tests {
    use std::{env, hint::black_box, path::PathBuf, time::Instant};

    use models::layout::{DecoderConfig, ModelLayout};

    use super::*;
    use crate::engine::{Error, ModelTensors, Stream};

    #[test]
    #[ignore = "benchmark; set MIRMIR_BENCH_MODEL or MODEL"]
    fn bench_hybrid_moe_components() -> Result<()> {
        let root = model_root()?;
        let layout = ModelLayout::inspect(&root)?;
        let decoder = DecoderConfig::from_layout(&layout)?;
        let load_stream = Stream::new_cpu()?;
        let tensors = ModelTensors::load(root, &load_stream)?;
        let stream = Stream::new_gpu()?;
        let index = env_usize("MIRMIR_BENCH_LAYER", 0)?;
        let config = HybridMoeLayerConfig::from_decoder(index, &decoder, 64)?;
        let weights = LayerWeights::load(&tensors, config, &stream)?;
        let input = Array::from_f32(&vec![0.25; decoder.hidden_size], &[1, 1, config.hidden_size])?
            .astype_like(&weights.router.expert_scale, &stream)?;
        let iters = env_usize("MIRMIR_BENCH_ITERS", 12)?;
        let warmup = env_usize("MIRMIR_BENCH_WARMUP", 3)?;
        let router_ms =
            measure(iters, warmup, &stream, || router_logits(&input, &weights, config, &stream))?;
        let routing_ms = measure(iters, warmup, &stream, || {
            Ok(routing(&input, &weights, config, &stream)?.indices)
        })?;
        let route = routing(&input, &weights, config, &stream)?;
        route.indices.async_eval()?;
        route.weights.async_eval()?;
        stream.synchronize()?;
        let normalized = weights.pre_expert_norm.apply(&input, config.rms_norm_eps, &stream)?;
        let expanded = normalized.expand_dims(&[-2, -3], &stream)?;
        let gate_up_ms = measure(iters, warmup, &stream, || {
            Ok(weights.experts.gather_gate_up(&expanded, &route.indices, false, &stream)?.0)
        })?;
        let (gate, up) =
            weights.experts.gather_gate_up(&expanded, &route.indices, false, &stream)?;
        let activated = gate.gelu_approx_mul(&up, &stream)?;
        let down_ms = measure(iters, warmup, &stream, || {
            weights.experts.down.gather(&activated, &route.indices, false, &stream)
        })?;
        let dense_ms =
            measure(iters, warmup, &stream, || dense(&input, &weights, config, None, &stream))?;
        let full_moe_ms =
            measure(iters, warmup, &stream, || experts(&input, &weights, config, None, &stream))?;
        let full_feed_forward_ms = measure(iters, warmup, &stream, || {
            forward(&input, &weights, config, None, None, &stream)
        })?;
        println!(
            "hybrid_moe_bench layer={index} iters={iters} warmup={warmup} prepared_gate_up=0 router_ms={router_ms:.4} routing_ms={routing_ms:.4} gate_up_ms={gate_up_ms:.4} down_ms={down_ms:.4} dense_ms={dense_ms:.4} full_moe_ms={full_moe_ms:.4} full_feed_forward_ms={full_feed_forward_ms:.4}"
        );
        Ok(())
    }

    fn router_logits(
        input: &Array,
        weights: &LayerWeights,
        config: HybridMoeLayerConfig,
        stream: &Stream,
    ) -> Result<Array> {
        let input = input.rms_norm(&weights.router.norm_scale, config.rms_norm_eps, stream)?;
        weights.router.projection.forward(&input, stream)
    }

    fn routing(
        input: &Array,
        weights: &LayerWeights,
        config: HybridMoeLayerConfig,
        stream: &Stream,
    ) -> Result<crate::engine::RouterOutput> {
        router_logits(input, weights, config, stream)?.router_top_k(
            &weights.router.expert_scale,
            config.top_k,
            stream,
        )
    }

    fn measure(
        iters: usize,
        warmup: usize,
        stream: &Stream,
        mut run: impl FnMut() -> Result<Array>,
    ) -> Result<f64> {
        for _ in 0..warmup {
            let output = run()?;
            output.async_eval()?;
            stream.synchronize()?;
            black_box(output);
        }
        let started = Instant::now();
        for _ in 0..iters {
            let output = run()?;
            output.async_eval()?;
            stream.synchronize()?;
            black_box(output);
        }
        let iters = iters.to_string().parse::<f64>()?;
        Ok(started.elapsed().as_secs_f64() * 1_000.0 / iters)
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
}
