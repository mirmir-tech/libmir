//! The attention projections of a dense layer: its own keys and values, or
//! only queries over an earlier layer's (Gemma 4's E models).

use models::weights::{DenseSoftmaxBindings, SharedKeyValueLayerBindings, TensorBinding};

use crate::{Error, Result};

#[derive(Debug, Clone, Copy)]
pub enum DenseAttentionBindings<'a> {
    /// Projects, normalizes and caches its own keys and values.
    Projected(DenseSoftmaxBindings<'a>),
    /// Attends over an earlier layer's cached keys and values: it projects
    /// queries only.
    Shared {
        query: &'a TensorBinding,
        output: &'a TensorBinding,
        query_norm: Option<&'a TensorBinding>,
    },
}

impl<'a> DenseAttentionBindings<'a> {
    pub(in crate::checkpoint) fn shared(bindings: &SharedKeyValueLayerBindings<'a>) -> Self {
        Self::Shared {
            query: bindings.query,
            output: bindings.output,
            query_norm: bindings.query_norm,
        }
    }

    #[must_use]
    pub const fn query(self) -> &'a TensorBinding {
        match self {
            Self::Projected(attention) => attention.query,
            Self::Shared { query, .. } => query,
        }
    }

    #[must_use]
    pub const fn output(self) -> &'a TensorBinding {
        match self {
            Self::Projected(attention) => attention.output,
            Self::Shared { output, .. } => output,
        }
    }

    #[must_use]
    pub const fn query_norm(self) -> Option<&'a TensorBinding> {
        match self {
            Self::Projected(attention) => attention.query_norm,
            Self::Shared { query_norm, .. } => query_norm,
        }
    }

    #[must_use]
    pub const fn key_norm(self) -> Option<&'a TensorBinding> {
        match self {
            Self::Projected(attention) => attention.key_norm,
            Self::Shared { .. } => None,
        }
    }

    /// The query, key, value and output projections the layer has.
    #[must_use]
    pub fn projections(self) -> Vec<&'a TensorBinding> {
        match self {
            Self::Projected(attention) => {
                vec![attention.query, attention.key, attention.value, attention.output]
            },
            Self::Shared { query, output, .. } => vec![query, output],
        }
    }

    /// The layer's own key and value projections, which loaders of formats
    /// without query-only layers require.
    pub fn projected(self) -> Result<DenseSoftmaxBindings<'a>> {
        match self {
            Self::Projected(attention) => Ok(attention),
            Self::Shared { .. } => Err(Error::UnsupportedDecoderLayer(
                "CUDA shares keys and values between GGML or BF16 layers only".into(),
            )),
        }
    }
}
