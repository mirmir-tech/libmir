use crate::engine::{KvPageFormat, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(in crate::engine::kv::paged) struct ArenaKey {
    pub(super) layer: usize,
    pub(super) page_size: usize,
    pub(super) kv_heads: usize,
    pub(super) head_dim: usize,
    pub(super) stored_dim: usize,
    pub(super) dtype: mirtal::DType,
    pub(super) format: KvPageFormat,
}

impl ArenaKey {
    pub(in crate::engine::kv::paged) fn new(
        layer: usize,
        page_size: usize,
        format: KvPageFormat,
        kv_heads: usize,
        head_dim: usize,
        input_dtype: mirtal::DType,
    ) -> Result<Self> {
        Ok(Self {
            layer,
            page_size,
            format,
            kv_heads,
            head_dim,
            stored_dim: format.packed_words(head_dim)?,
            dtype: if format.quantized() {
                mirtal::DType::Uint32
            } else {
                input_dtype
            },
        })
    }
}
