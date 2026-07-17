use models::layout::AttentionLayerType;

use crate::engine::{Error, GatedDeltaState, KvCache, Result};

#[derive(Debug)]
pub(super) enum HybridLinearLayerCache {
    Linear(GatedDeltaState),
    Full(KvCache),
}

pub(super) fn new(
    layer_types: &[AttentionLayerType],
    step: usize,
) -> Result<Vec<HybridLinearLayerCache>> {
    layer_types
        .iter()
        .map(|layer_type| match layer_type {
            AttentionLayerType::Linear => {
                GatedDeltaState::new().map(HybridLinearLayerCache::Linear)
            },
            AttentionLayerType::Full => {
                KvCache::new_paged(step, 16).map(HybridLinearLayerCache::Full)
            },
            AttentionLayerType::Sliding => KvCache::new(step).map(HybridLinearLayerCache::Full),
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
