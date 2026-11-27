//! What one layer's decode attended over, for the later layers that attend
//! over the same keys and values without keeping their own (Gemma 4's E
//! models).

use std::sync::Arc;

use super::{History, attend_over, row_attention};
use crate::engine::{Array, KvContext, Result, Stream};

#[derive(Debug)]
pub enum Attended {
    /// The persistent history every row attended over together.
    History(Arc<History>),
    /// Each row's cache context.
    Rows(Vec<KvContext>),
}

impl Attended {
    /// `queries`, rotated `[B, H, 1, D]`, attended over the same keys and
    /// values.
    pub fn attend(
        &self,
        queries: &Array,
        scale: f32,
        sinks: Option<&Array>,
        stream: &Stream,
    ) -> Result<Array> {
        match self {
            Self::History(history) => attend_over(history, queries, scale, sinks, stream),
            Self::Rows(contexts) => row_attention(queries, contexts, scale, sinks, stream),
        }
    }
}
