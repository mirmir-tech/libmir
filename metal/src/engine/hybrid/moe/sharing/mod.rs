//! What one forward step hands between the layers of Gemma 4's E models:
//! every layer's per-layer input, and the keys and values of the layers that
//! later layers attend over instead of projecting their own.
//!
//! The model's layers run in order on its one worker, so the model sets the
//! per-layer inputs and a source layer publishes its keys and values before
//! any layer reads them in the same step.

mod per_layer;

use std::sync::{Mutex, PoisonError};

pub(super) use per_layer::{PerLayerGate, PerLayerInputs};

use crate::engine::{Array, Error, Result, Stream};

#[derive(Debug, Default)]
pub(super) struct StepShare {
    /// `[B, S, layers, width]` per-layer inputs of the current step.
    per_layer: Mutex<Option<Array>>,
    /// Normalized, unrotated `[B, S, kv_heads, D]` keys and `[B, kv_heads,
    /// S, D]` values each source layer computed in the current step.
    key_values: Mutex<Vec<Option<(Array, Array)>>>,
}

impl StepShare {
    pub(super) fn new(layers: usize) -> Self {
        Self {
            per_layer: Mutex::new(None),
            key_values: Mutex::new((0..layers).map(|_| None).collect()),
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

    pub(super) fn publish(&self, layer: usize, keys: &Array, values: &Array) -> Result<()> {
        let entry = Some((keys.snapshot()?, values.snapshot()?));
        let mut slots = self.key_values.lock().unwrap_or_else(PoisonError::into_inner);
        let slot = slots
            .get_mut(layer)
            .ok_or_else(|| Error::InvalidModel(format!("no K/V slot for layer {layer}")))?;
        *slot = entry;
        drop(slots);
        Ok(())
    }

    /// The keys and values layer `source` published in the current step.
    pub(super) fn key_values(&self, source: usize) -> Result<(Array, Array)> {
        let slots = self.key_values.lock().unwrap_or_else(PoisonError::into_inner);
        let pair = slots
            .get(source)
            .and_then(Option::as_ref)
            .map(|(keys, values)| Ok::<_, Error>((keys.snapshot()?, values.snapshot()?)))
            .transpose()?;
        drop(slots);
        pair.ok_or_else(|| {
            Error::InvalidModel(format!("layer {source} published no keys and values"))
        })
    }
}
