use crate::{
    PrefillDenseSwiGlu, PrefillMoeBlockBf16,
    backend::{block::PreparedDecodeMoeBlock, dense::graph::PreparedDecodeDense},
};

pub(in crate::backend::model) enum PreparedLayer {
    Moe(Box<PreparedDecodeMoeBlock>),
    Dense(Box<PreparedDecodeDense>),
}

impl PreparedLayer {
    pub const fn layer(&self) -> usize {
        match self {
            Self::Moe(prepared) => prepared.layer,
            Self::Dense(prepared) => prepared.block.attention.config.layer,
        }
    }
}

pub(in crate::backend::model) enum LayerPrefill<'a> {
    Moe(&'a mut PrefillMoeBlockBf16),
    Dense(&'a mut PrefillDenseSwiGlu),
}
