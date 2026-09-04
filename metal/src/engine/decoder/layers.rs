use std::time::{Duration, Instant};

use super::{Array, DecoderCache, ImageTokenSpan, Result, Stream};

const PREFILL_UNSEGMENTED_TOKEN_PAIRS: usize = 4 * 1_024 * 1_024;
const PREFILL_COMMAND_LAYER_TOKEN_PAIRS: usize = 96 * 1_024 * 1_024;
const LONG_PREFILL_CONTEXT: usize = 16 * 1_024;
const LONG_PREFILL_MINIMUM_QUERY: usize = 512;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MixerKind {
    Softmax,
    Linear,
}

impl MixerKind {
    const fn name(self) -> &'static str {
        match self {
            Self::Softmax => "softmax",
            Self::Linear => "linear",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct LayerContext<'a> {
    pub position: i32,
    pub causal: bool,
    pub positions: Option<&'a Array>,
    pub image: Option<ImageTokenSpan>,
    pub stream: &'a Stream,
}

#[derive(Debug, Clone, Copy)]
pub struct LayerLoopOptions {
    profile_layers: bool,
    evaluation_step: Option<usize>,
    profile_graph_build: bool,
}

impl LayerLoopOptions {
    pub const fn new(
        profile_layers: bool,
        evaluation_step: Option<usize>,
        profile_graph_build: bool,
    ) -> Self {
        Self {
            profile_layers,
            evaluation_step,
            profile_graph_build,
        }
    }

    pub const fn disabled() -> Self {
        Self::new(false, None, false)
    }
}

pub trait LoweredLayer {
    fn mixer_kind(&self) -> MixerKind;

    fn forward_mixer(
        &self,
        input: &Array,
        cache: &mut DecoderCache,
        index: usize,
        context: LayerContext<'_>,
    ) -> Result<Array>;

    fn forward_feed_forward(&self, input: &Array, context: LayerContext<'_>) -> Result<Array>;
}

pub trait LoweredPackedLayer {
    fn forward_packed_mixer(
        &self,
        input: &Array,
        caches: &mut [&mut DecoderCache],
        index: usize,
        positions: &[i32],
        stream: &Stream,
    ) -> Result<Array>;

    fn forward_packed_feed_forward(
        &self,
        input: &Array,
        batch_size: usize,
        stream: &Stream,
    ) -> Result<Array>;
}

pub fn forward_layers<L: LoweredLayer>(
    layers: &[L],
    mut hidden: Array,
    cache: &mut DecoderCache,
    context: LayerContext<'_>,
    options: LayerLoopOptions,
) -> Result<Array> {
    let graph_started = Instant::now();
    let mut linear_graph = Duration::ZERO;
    let mut softmax_graph = Duration::ZERO;
    for (index, layer) in layers.iter().enumerate() {
        let started = Instant::now();
        hidden = layer.forward_mixer(&hidden, cache, index, context)?;
        hidden = layer.forward_feed_forward(&hidden, context)?;
        let elapsed = started.elapsed();
        if options.profile_graph_build {
            match layer.mixer_kind() {
                MixerKind::Linear => linear_graph += elapsed,
                MixerKind::Softmax => softmax_graph += elapsed,
            }
        }
        let evaluate = options.profile_layers
            || options.evaluation_step.is_some_and(|step| {
                step != 0 && ((index + 1) % step == 0 || index + 1 == layers.len())
            });
        if evaluate {
            hidden.async_eval(context.stream)?;
            context.stream.synchronize()?;
            if options.evaluation_step.is_some() {
                hidden.detach_graph(context.stream)?;
                context.stream.detach_paged_arena_graphs()?;
                crate::engine::clear_memory_cache()?;
            }
        }
        if options.profile_layers {
            tracing::debug!(
                layer = index,
                mixer = layer.mixer_kind().name(),
                milliseconds = elapsed.as_secs_f64() * 1_000.0,
                "MLX lowered decoder layer profile"
            );
        }
    }
    if options.profile_graph_build {
        let sequence = hidden.shape()?.get(1).copied().unwrap_or_default();
        tracing::debug!(
            sequence,
            total_ms = graph_started.elapsed().as_secs_f64() * 1_000.0,
            linear_ms = linear_graph.as_secs_f64() * 1_000.0,
            softmax_ms = softmax_graph.as_secs_f64() * 1_000.0,
            "MLX lowered decoder graph-build profile"
        );
    }
    Ok(hidden)
}

pub fn forward_packed_layers<L: LoweredPackedLayer>(
    layers: &[L],
    mut hidden: Array,
    caches: &mut [&mut DecoderCache],
    positions: &[i32],
    stream: &Stream,
) -> Result<Array> {
    for (index, layer) in layers.iter().enumerate() {
        hidden = layer.forward_packed_mixer(&hidden, caches, index, positions, stream)?;
        hidden = layer.forward_packed_feed_forward(&hidden, caches.len(), stream)?;
    }
    Ok(hidden)
}

pub(in crate::engine) fn prefill_evaluation_step(
    batch: usize,
    sequence: usize,
    position: usize,
    layers: usize,
) -> Option<usize> {
    let context = position.saturating_add(sequence);
    let planned_sequence = if context >= LONG_PREFILL_CONTEXT {
        sequence.max(LONG_PREFILL_MINIMUM_QUERY)
    } else {
        sequence
    };
    let work = batch.saturating_mul(planned_sequence).saturating_mul(context).max(1);
    if work <= PREFILL_UNSEGMENTED_TOKEN_PAIRS {
        return None;
    }
    let step = (PREFILL_COMMAND_LAYER_TOKEN_PAIRS / work).max(1);
    (step < layers).then_some(step)
}

#[cfg(test)]
mod tests {
    use super::prefill_evaluation_step;

    #[test]
    fn segments_only_expensive_prefill_graphs() {
        assert_eq!(prefill_evaluation_step(1, 512, 0, 36), None);
        assert_eq!(prefill_evaluation_step(1, 512, 7_680, 36), None);
        assert_eq!(prefill_evaluation_step(1, 512, 8_192, 36), Some(22));
        assert_eq!(prefill_evaluation_step(10, 512, 8_192, 36), Some(2));
        assert_eq!(prefill_evaluation_step(1, 2, 32_767, 36), Some(5));
    }
}
