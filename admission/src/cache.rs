//! Per-layer cache layout emitted by architecture admission and consumed by
//! memory estimation, CUDA allocation, and Metal cache construction.

use std::num::NonZeroUsize;

use models::semantic::{
    AttentionWindow, DecoderLayerSpec, DecoderSpec, LayerCacheShape, MixerSpec,
};

use crate::cuda::CudaDecoderRuntime;

/// Sliding-ring physical slots per concurrent session on CUDA clamped-routed
/// models (one live slot plus three prefix-checkpoint slots).
pub const CUDA_SLIDING_SLOTS_PER_SESSION: usize = 4;

/// Which recurrent state a non-KV layer maintains per session.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecurrentCacheKind {
    Linear,
    StateSpace,
}

/// Physical cache storage admitted for one decoder layer on a target backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerCacheKind {
    /// Paged K/V blocks indexed by the runtime block pool.
    Paged { shape: LayerCacheShape },
    /// Bounded sliding ring outside the shared block pool.
    Ring {
        shape: LayerCacheShape,
        window: NonZeroUsize,
        slots_per_session: usize,
    },
    /// Reads an earlier layer's K/V cache (holds no storage of its own).
    SharedFrom { source_layer: usize },
    /// Fixed-size recurrent state per session (no K/V pages).
    Recurrent(RecurrentCacheKind),
}

/// One layer's admitted cache entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayerCacheEntry {
    pub layer: usize,
    pub kind: LayerCacheKind,
}

/// Single per-layer cache description shared across memory estimation and
/// backend allocators.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CacheLayout {
    pub layers: Vec<LayerCacheEntry>,
    pub prediction_layers: Vec<LayerCacheEntry>,
}

impl CacheLayout {
    /// Admitted Metal cache layout for `spec`.
    ///
    /// Metal keeps only `AttentionWindow::Sliding` layers in bounded rings;
    /// global attention, `AttentionWindow::Chunked`, and latent attention use
    /// paged K/V blocks.
    #[must_use]
    pub fn for_metal(spec: &DecoderSpec) -> Self {
        Self::build(spec, Some(1))
    }

    /// Admitted CUDA cache layout for `spec` under `runtime`.
    ///
    /// Only `CudaDecoderRuntime::ClampedRouted` keeps
    /// `AttentionWindow::Sliding` layers in bounded rings; all other
    /// attention and chunked layers use paged K/V blocks.
    #[must_use]
    pub fn for_cuda(spec: &DecoderSpec, runtime: CudaDecoderRuntime) -> Self {
        let sliding_slots = (runtime == CudaDecoderRuntime::ClampedRouted)
            .then_some(CUDA_SLIDING_SLOTS_PER_SESSION);
        Self::build(spec, sliding_slots)
    }

    fn build(spec: &DecoderSpec, sliding_slots: Option<usize>) -> Self {
        let layers = spec
            .layers
            .iter()
            .enumerate()
            .map(|(layer, spec)| LayerCacheEntry {
                layer,
                kind: layer_kind(spec, sliding_slots),
            })
            .collect();
        let base = spec.layers.len();
        let prediction_layers = spec
            .prediction_layers
            .iter()
            .enumerate()
            .map(|(offset, spec)| LayerCacheEntry {
                layer: base + offset,
                kind: layer_kind(spec, sliding_slots),
            })
            .collect();
        Self { layers, prediction_layers }
    }

    /// Whether any decoder layer stores recurrent (linear or state-space)
    /// state.
    #[must_use]
    pub fn has_recurrent_layers(&self) -> bool {
        self.layers
            .iter()
            .any(|entry| matches!(entry.kind, LayerCacheKind::Recurrent(_)))
    }

    /// Whether any layer holds state outside the shared paged K/V pool (sliding
    /// rings, recurrent states, or speculative prediction layers).
    #[must_use]
    pub fn has_non_paged_state(&self) -> bool {
        !self.prediction_layers.is_empty()
            || self.layers.iter().any(|entry| {
                matches!(entry.kind, LayerCacheKind::Ring { .. } | LayerCacheKind::Recurrent(_))
            })
    }

    /// Largest sliding-ring window across decoder layers, if any layer uses a
    /// ring.
    #[must_use]
    pub fn max_ring_window(&self) -> Option<usize> {
        self.layers
            .iter()
            .filter_map(|entry| match entry.kind {
                LayerCacheKind::Ring { window, .. } => Some(window.get()),
                LayerCacheKind::Paged { .. }
                | LayerCacheKind::SharedFrom { .. }
                | LayerCacheKind::Recurrent(_) => None,
            })
            .max()
    }

    /// Every decoder layer with its own paged K/V storage.
    pub fn paged_layers(&self) -> impl Iterator<Item = (usize, LayerCacheShape)> + '_ {
        self.layers.iter().filter_map(|entry| match entry.kind {
            LayerCacheKind::Paged { shape } => Some((entry.layer, shape)),
            LayerCacheKind::Ring { .. }
            | LayerCacheKind::SharedFrom { .. }
            | LayerCacheKind::Recurrent(_) => None,
        })
    }

    /// Every decoder layer that attends over a full paged history, including
    /// layers that share an earlier paged layer's K/V cache.
    pub fn paged_attention_layers(&self) -> impl Iterator<Item = (usize, LayerCacheShape)> + '_ {
        self.layers.iter().filter_map(|entry| match entry.kind {
            LayerCacheKind::Paged { shape } => Some((entry.layer, shape)),
            LayerCacheKind::SharedFrom { source_layer } => {
                match self.layers.get(source_layer).map(|source| source.kind) {
                    Some(LayerCacheKind::Paged { shape }) => Some((entry.layer, shape)),
                    _ => None,
                }
            },
            LayerCacheKind::Ring { .. } | LayerCacheKind::Recurrent(_) => None,
        })
    }
}

fn layer_kind(spec: &DecoderLayerSpec, sliding_slots: Option<usize>) -> LayerCacheKind {
    match &spec.mixer {
        MixerSpec::SoftmaxAttention(attention) => {
            if let Some(source_layer) = attention.key_value_source {
                return LayerCacheKind::SharedFrom { source_layer };
            }
            let shape = LayerCacheShape {
                heads: attention.key_value_heads,
                key_width: attention.head_dim,
                value_width: attention.head_dim,
            };
            match (attention.window, sliding_slots) {
                (Some(AttentionWindow::Sliding(window)), Some(slots_per_session)) => {
                    LayerCacheKind::Ring { shape, window, slots_per_session }
                },
                (None | Some(AttentionWindow::Chunked { .. } | AttentionWindow::Sliding(_)), _) => {
                    LayerCacheKind::Paged { shape }
                },
            }
        },
        MixerSpec::LatentAttention(latent) => LayerCacheKind::Paged {
            shape: LayerCacheShape {
                heads: 1,
                key_width: latent.cached_width(),
                value_width: 0,
            },
        },
        MixerSpec::LinearAttention(_) => LayerCacheKind::Recurrent(RecurrentCacheKind::Linear),
        MixerSpec::StateSpace(_) => LayerCacheKind::Recurrent(RecurrentCacheKind::StateSpace),
    }
}
