//! What one forward step hands between the layers of Gemma 4's E models:
//! every layer's per-layer input, and what each source layer attended over,
//! which the later layers sharing its keys and values attend over too
//! instead of keeping their own cache.
//!
//! The model's layers run in order on its one worker, so the model sets the
//! per-layer inputs and a source layer publishes what it attended over
//! before any layer reads it in the same step.

mod per_layer;
#[cfg(test)]
mod tests;

use std::sync::{Arc, Mutex, PoisonError};

pub(super) use per_layer::{PerLayerGate, PerLayerInputs};

use crate::engine::{
    Array, Error, KvContext, PagedContextMode, Result, Stream, persistent_history::Attended,
};

/// What a source layer attended over in the current step.
#[derive(Debug)]
pub(super) enum SourceAttention {
    /// Each sequence's cache context and the mode it was read in.
    Rows(Vec<(KvContext, PagedContextMode)>),
    /// A packed decode's persistent history, or its rows.
    Persistent(Attended),
}

#[derive(Debug, Default)]
pub(super) struct StepShare {
    /// `[B, S, layers, width]` per-layer inputs of the current step.
    per_layer: Mutex<Option<Array>>,
    /// What each source layer attended over in the current step, held only
    /// until the last layer sharing it has read it.
    attended: Mutex<Vec<Slot>>,
}

#[derive(Debug, Default)]
struct Slot {
    attended: Option<Arc<SourceAttention>>,
    /// The layers sharing this one's keys and values.
    readers: usize,
    /// Of them, those yet to read the current step's attention.
    unread: usize,
}

impl StepShare {
    /// A step of a model whose layer `i` is read by `readers[i]` later
    /// layers sharing its keys and values.
    pub(super) fn new(readers: &[usize]) -> Self {
        Self {
            per_layer: Mutex::new(None),
            attended: Mutex::new(
                readers.iter().map(|&readers| Slot { readers, ..Slot::default() }).collect(),
            ),
        }
    }

    pub(super) fn set_per_layer(&self, inputs: Array) {
        *self.per_layer.lock().unwrap_or_else(PoisonError::into_inner) = Some(inputs);
    }

    /// Layer `index`'s `[B, S, width]` input in the current step.
    pub(super) fn per_layer(&self, index: usize, stream: &Stream) -> Result<Array> {
        let inputs = self
            .per_layer
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .as_ref()
            .map(Array::snapshot)
            .transpose()?
            .ok_or_else(|| Error::InvalidModel("per-layer inputs were not prepared".into()))?;
        let shape = inputs.shape()?;
        let [batch, sequence, _, width] = *shape.as_slice() else {
            return Err(Error::InvalidModel("per-layer inputs are not [B, S, L, W]".into()));
        };
        let (rows, tokens, lanes) =
            (usize::try_from(batch)?, usize::try_from(sequence)?, usize::try_from(width)?);
        inputs
            .slice(&[0, 0, index, 0], &[rows, tokens, index + 1, lanes], stream)?
            .reshape(&[batch, sequence, width], stream)
    }

    pub(super) fn publish(&self, layer: usize, attended: SourceAttention) -> Result<()> {
        let mut slots = self.attended.lock().unwrap_or_else(PoisonError::into_inner);
        let slot = slots
            .get_mut(layer)
            .ok_or_else(|| Error::InvalidModel(format!("no attention slot for layer {layer}")))?;
        slot.attended = (slot.readers > 0).then(|| Arc::new(attended));
        slot.unread = slot.readers;
        drop(slots);
        Ok(())
    }

    /// What layer `source` attended over in the current step. The last
    /// layer to read it takes it, so its keys and values are freed with that
    /// layer's attention rather than held into the next step.
    pub(super) fn attended(&self, source: usize) -> Result<Arc<SourceAttention>> {
        let mut slots = self.attended.lock().unwrap_or_else(PoisonError::into_inner);
        let attended = slots.get_mut(source).and_then(|slot| {
            slot.unread = slot.unread.checked_sub(1)?;
            if slot.unread == 0 {
                slot.attended.take()
            } else {
                slot.attended.as_ref().map(Arc::clone)
            }
        });
        drop(slots);
        attended.ok_or_else(|| {
            Error::InvalidModel(format!("layer {source} published no attention for this step"))
        })
    }
}
