use std::sync::Arc;

use architecture::{LayerCacheEntry, LayerCacheKind, RecurrentCacheKind};

use crate::engine::{
    Error, GatedDeltaState, KvCache, KvPageFormat, PagedArenaPool, Result, Stream,
    state_space_hybrid::StateSpaceState,
};

#[derive(Debug)]
#[allow(clippy::large_enum_variant)] // Session-local K/V stays inline; boxing adds per-layer allocation.
pub(super) enum HybridLinearLayerCache {
    Linear(GatedDeltaState),
    StateSpace(StateSpaceState),
    Full(KvCache),
}

pub(super) fn from_entries(
    entries: &[LayerCacheEntry],
    step: usize,
    format: KvPageFormat,
    page_size: usize,
    max_pages: usize,
    pool: &Arc<PagedArenaPool>,
) -> Result<Vec<HybridLinearLayerCache>> {
    entries
        .iter()
        .map(|entry| match entry.kind {
            LayerCacheKind::Recurrent(RecurrentCacheKind::Linear) => {
                GatedDeltaState::new().map(HybridLinearLayerCache::Linear)
            },
            LayerCacheKind::Recurrent(RecurrentCacheKind::StateSpace) => {
                Ok(HybridLinearLayerCache::StateSpace(StateSpaceState::new()))
            },
            LayerCacheKind::Paged { .. } => KvCache::new_paged_with_pool_capacity(
                step,
                page_size,
                format,
                max_pages,
                Arc::clone(pool),
                entry.layer,
            )
            .map(HybridLinearLayerCache::Full),
            LayerCacheKind::Ring { window, .. } => {
                KvCache::new_with_window(step, Some(window.get())).map(HybridLinearLayerCache::Full)
            },
            LayerCacheKind::SharedFrom { .. } => Err(Error::InvalidModel(format!(
                "hybrid cache layer {} cannot share an earlier layer's K/V",
                entry.layer
            ))),
        })
        .collect()
}

pub(super) fn reset(layers: &mut [HybridLinearLayerCache]) -> Result<()> {
    for layer in layers {
        match layer {
            HybridLinearLayerCache::Linear(state) => state.reset()?,
            HybridLinearLayerCache::StateSpace(state) => state.reset(),
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

pub(super) fn bind_blocks(
    layers: &mut [HybridLinearLayerCache],
    blocks: &[runtime::kv::BlockId],
    token_len: Option<usize>,
    stream: &Stream,
) -> Result<()> {
    for layer in layers {
        if let HybridLinearLayerCache::Full(cache) = layer {
            cache.bind_blocks(blocks, token_len, stream)?;
        }
    }
    Ok(())
}

pub(super) fn can_write_token(layers: &[HybridLinearLayerCache], token_pos: usize) -> bool {
    layers.iter().all(|layer| match layer {
        HybridLinearLayerCache::Full(cache) => cache.can_write_token(token_pos),
        HybridLinearLayerCache::Linear(_) | HybridLinearLayerCache::StateSpace(_) => true,
    })
}

pub(super) fn detach_evaluated_graphs(
    layers: &[HybridLinearLayerCache],
    stream: &Stream,
) -> Result<()> {
    for layer in layers {
        match layer {
            HybridLinearLayerCache::Linear(state) => state.detach_evaluated_graphs(stream)?,
            HybridLinearLayerCache::StateSpace(state) => state.detach_evaluated_graphs(stream)?,
            HybridLinearLayerCache::Full(cache) => cache.detach_evaluated_graphs(stream)?,
        }
    }
    Ok(())
}

pub(super) fn graph_roots(layers: &[HybridLinearLayerCache]) -> Vec<&crate::engine::Array> {
    layers
        .iter()
        .flat_map(|layer| -> Vec<&crate::engine::Array> {
            match layer {
                HybridLinearLayerCache::Linear(state) => state.graph_roots().collect(),
                HybridLinearLayerCache::StateSpace(state) => state.graph_roots().collect(),
                HybridLinearLayerCache::Full(_) => Vec::new(),
            }
        })
        .collect()
}

pub(super) fn gated_delta_state(
    layers: &mut [HybridLinearLayerCache],
    index: usize,
) -> Result<&mut GatedDeltaState> {
    match layers.get_mut(index) {
        Some(HybridLinearLayerCache::Linear(state)) => Ok(state),
        Some(HybridLinearLayerCache::Full(_) | HybridLinearLayerCache::StateSpace(_)) => {
            Err(Error::InvalidModel(format!("layer {index} requires a linear attention state")))
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
        Some(HybridLinearLayerCache::Linear(_) | HybridLinearLayerCache::StateSpace(_)) => {
            Err(Error::InvalidModel(format!("layer {index} requires a recurrent state")))
        },
        None => Err(Error::InvalidModel(format!("missing hybrid cache layer {index}"))),
    }
}

pub(super) fn state_space_state(
    layers: &mut [HybridLinearLayerCache],
    index: usize,
) -> Result<&mut StateSpaceState> {
    match layers.get_mut(index) {
        Some(HybridLinearLayerCache::StateSpace(state)) => Ok(state),
        Some(_) => Err(Error::InvalidModel(format!("layer {index} is not a state-space layer"))),
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
        HybridLinearLayerCache::StateSpace(state) => Ok(state.offset()),
        HybridLinearLayerCache::Full(cache) => cache.offset(),
    }
}

fn snapshot(layer: &HybridLinearLayerCache) -> Result<HybridLinearLayerCache> {
    match layer {
        HybridLinearLayerCache::Linear(state) => {
            state.snapshot().map(HybridLinearLayerCache::Linear)
        },
        HybridLinearLayerCache::StateSpace(state) => {
            state.snapshot().map(HybridLinearLayerCache::StateSpace)
        },
        HybridLinearLayerCache::Full(cache) => {
            cache.snapshot_at(cache.offset()?).map(HybridLinearLayerCache::Full)
        },
    }
}
