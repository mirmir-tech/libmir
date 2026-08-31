use std::sync::Arc;

use crate::engine::{
    Error, GatedDeltaState, KvCache, KvPageFormat, PagedArenaPool, Result, Stream,
    lowering::MixerLowering,
};

#[derive(Debug)]
#[allow(clippy::large_enum_variant)] // Session-local K/V stays inline; boxing adds per-layer allocation.
pub(super) enum HybridLinearLayerCache {
    Linear(GatedDeltaState),
    Full(KvCache),
}

pub(super) fn new(
    mixers: &[MixerLowering],
    step: usize,
    format: KvPageFormat,
    page_size: usize,
    max_pages: usize,
    pool: &Arc<PagedArenaPool>,
) -> Result<Vec<HybridLinearLayerCache>> {
    mixers
        .iter()
        .enumerate()
        .map(|(layer, mixer)| match mixer {
            MixerLowering::Linear => GatedDeltaState::new().map(HybridLinearLayerCache::Linear),
            MixerLowering::Softmax { window: None, .. } => KvCache::new_paged_with_pool_capacity(
                step,
                page_size,
                format,
                max_pages,
                Arc::clone(pool),
                layer,
            )
            .map(HybridLinearLayerCache::Full),
            MixerLowering::Softmax { window: Some(window), .. } => {
                KvCache::new_with_window(step, Some(*window)).map(HybridLinearLayerCache::Full)
            },
        })
        .collect()
}

pub(super) fn reset(layers: &mut [HybridLinearLayerCache]) -> Result<()> {
    for layer in layers {
        match layer {
            HybridLinearLayerCache::Linear(state) => state.reset()?,
            HybridLinearLayerCache::Full(cache) => cache.reset()?,
        }
    }
    Ok(())
}

pub(super) fn reserve(layers: &mut [HybridLinearLayerCache], tokens: usize) -> Result<()> {
    for layer in layers {
        if let HybridLinearLayerCache::Full(cache) = layer {
            cache.reserve(tokens)?;
        }
    }
    Ok(())
}

pub(super) fn plan_contiguous(layers: &mut [HybridLinearLayerCache], tokens: usize) {
    for layer in layers {
        if let HybridLinearLayerCache::Full(cache) = layer {
            cache.plan_contiguous(tokens);
        }
    }
}

pub(super) fn detach_evaluated_graphs(
    layers: &[HybridLinearLayerCache],
    stream: &Stream,
) -> Result<()> {
    for layer in layers {
        match layer {
            HybridLinearLayerCache::Linear(state) => state.detach_evaluated_graphs(stream)?,
            HybridLinearLayerCache::Full(cache) => cache.detach_evaluated_graphs(stream)?,
        }
    }
    Ok(())
}

pub(super) fn graph_roots(layers: &[HybridLinearLayerCache]) -> Vec<&crate::engine::Array> {
    layers
        .iter()
        .filter_map(|layer| match layer {
            HybridLinearLayerCache::Linear(state) => Some(state),
            HybridLinearLayerCache::Full(_) => None,
        })
        .flat_map(GatedDeltaState::graph_roots)
        .collect()
}

pub(super) fn gated_delta_state(
    layers: &mut [HybridLinearLayerCache],
    index: usize,
) -> Result<&mut GatedDeltaState> {
    match layers.get_mut(index) {
        Some(HybridLinearLayerCache::Linear(state)) => Ok(state),
        Some(HybridLinearLayerCache::Full(_)) => {
            Err(Error::InvalidModel(format!("layer {index} requires a full attention cache")))
        },
        None => Err(Error::InvalidModel(format!("missing hybrid cache layer {index}"))),
    }
}

pub(super) fn full_attention_cache(
    layers: &mut [HybridLinearLayerCache],
    index: usize,
) -> Result<&mut KvCache> {
    match layers.get_mut(index) {
        Some(HybridLinearLayerCache::Full(cache)) => Ok(cache),
        Some(HybridLinearLayerCache::Linear(_)) => {
            Err(Error::InvalidModel(format!("layer {index} requires a linear attention state")))
        },
        None => Err(Error::InvalidModel(format!("missing hybrid cache layer {index}"))),
    }
}

pub(super) fn offset(layers: &[HybridLinearLayerCache]) -> Result<usize> {
    let Some(first) = layers.first() else {
        return Ok(0);
    };
    let offset = layer_offset(first)?;
    for layer in layers {
        if layer_offset(layer)? != offset {
            return Err(Error::InvalidModel("hybrid cache layers have diverging offsets".into()));
        }
    }
    Ok(offset)
}

pub(super) fn snapshot_at(
    layers: &[HybridLinearLayerCache],
    requested_offset: usize,
) -> Result<Vec<HybridLinearLayerCache>> {
    if offset(layers)? != requested_offset {
        return Err(Error::InvalidModel(
            "hybrid linear cache can only snapshot its current recurrent state".into(),
        ));
    }
    layers.iter().map(snapshot).collect()
}

fn layer_offset(layer: &HybridLinearLayerCache) -> Result<usize> {
    match layer {
        HybridLinearLayerCache::Linear(state) => state.offset(),
        HybridLinearLayerCache::Full(cache) => cache.offset(),
    }
}

fn snapshot(layer: &HybridLinearLayerCache) -> Result<HybridLinearLayerCache> {
    match layer {
        HybridLinearLayerCache::Linear(state) => {
            state.snapshot().map(HybridLinearLayerCache::Linear)
        },
        HybridLinearLayerCache::Full(cache) => {
            cache.snapshot_at(cache.offset()?).map(HybridLinearLayerCache::Full)
        },
    }
}
