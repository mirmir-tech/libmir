use std::time::Instant;

use super::{Array, DecoderCache, MixerKind, Result, Stream};

pub(in crate::engine) enum Component {
    Mixer(MixerKind),
    FeedForward,
}

impl Component {
    const fn name(self) -> &'static str {
        match self {
            Self::Mixer(mixer) => mixer.name(),
            Self::FeedForward => "feed_forward",
        }
    }
}

pub(in crate::engine) fn record(
    started: Option<Instant>,
    hidden: &Array,
    caches: &[&mut DecoderCache],
    stream: &Stream,
    layer: usize,
    component: Component,
) -> Result<()> {
    let Some(started) = started else {
        return Ok(());
    };
    let graph_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let mut roots = vec![hidden];
    for cache in caches {
        cache.extend_graph_roots(&mut roots);
    }
    stream.eval_many_with_paged_arenas(&roots)?;
    stream.synchronize()?;
    tracing::debug!(
        layer,
        batch = caches.len(),
        component = component.name(),
        graph_ms,
        total_ms = started.elapsed().as_secs_f64() * 1_000.0,
        "Metal packed component profile"
    );
    Ok(())
}

/// Explicit diagnostic barriers; disabled execution adds no evaluation or sync.
pub(in crate::engine) enum PrefillProfile {
    Disabled,
    Active { _span: tracing::span::EnteredSpan },
}

impl PrefillProfile {
    pub fn begin(
        hidden: &Array,
        caches: &[&mut DecoderCache],
        positions: &[i32],
        stream: &Stream,
    ) -> Result<Self> {
        let config = &stream.config().diagnostics;
        let enabled = config.profile_components;
        #[cfg(test)]
        let enabled = config.prefill_component_window.as_ref().map_or(enabled, |window| {
            positions.iter().all(|&position| {
                usize::try_from(position).is_ok_and(|position| window.contains(&position))
            })
        });
        if !enabled {
            return Ok(Self::Disabled);
        }
        let sequence = hidden.shape()?.get(1).copied().unwrap_or_default();
        let span = tracing::debug_span!(
            "Metal packed prefill profile",
            ?positions,
            sequence,
            batch = caches.len()
        )
        .entered();
        // Settle embedding and prior work before charging the first mixer.
        let mut roots = vec![hidden];
        for cache in caches {
            cache.extend_graph_roots(&mut roots);
        }
        stream.eval_many_with_paged_arenas(&roots)?;
        stream.synchronize()?;
        Ok(Self::Active { _span: span })
    }

    pub fn start(&self) -> Option<Instant> {
        match self {
            Self::Disabled => None,
            Self::Active { .. } => Some(Instant::now()),
        }
    }
}
