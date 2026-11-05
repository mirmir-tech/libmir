//! The layer a typed binding view reads.

use std::fmt;

use super::{LayerTensorRole, LogicalTensorRole, PredictionTensorRole};

/// A decoder layer or a multi-token prediction module; both hold tensors
/// with decoder-layer roles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerSlot {
    Decoder(usize),
    Prediction(usize),
}

impl LayerSlot {
    /// The logical role of a decoder-layer tensor in this slot.
    #[must_use]
    pub fn role(self, tensor: LayerTensorRole) -> LogicalTensorRole {
        match self {
            Self::Decoder(index) => LogicalTensorRole::Layer { index, tensor },
            Self::Prediction(depth) => LogicalTensorRole::Prediction {
                depth,
                tensor: PredictionTensorRole::Layer { tensor },
            },
        }
    }

    /// The decoder-layer role of `role` when it belongs to this slot.
    #[must_use]
    pub fn layer_tensor(self, role: &LogicalTensorRole) -> Option<&LayerTensorRole> {
        match (self, role) {
            (Self::Decoder(layer), LogicalTensorRole::Layer { index, tensor })
                if *index == layer =>
            {
                Some(tensor)
            },
            (
                Self::Prediction(module),
                LogicalTensorRole::Prediction {
                    depth,
                    tensor: PredictionTensorRole::Layer { tensor },
                },
            ) if *depth == module => Some(tensor),
            _ => None,
        }
    }
}

impl fmt::Display for LayerSlot {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Decoder(index) => write!(formatter, "layer {index}"),
            Self::Prediction(depth) => write!(formatter, "prediction module {depth}"),
        }
    }
}
