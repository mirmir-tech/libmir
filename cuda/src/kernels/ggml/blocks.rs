//! GGML block storage the kernels may read a little past.

use mircuda::{DeviceBuffer, MemoryPool, Stream};

use crate::Result;

/// GGML blocks whose allocation runs [`Self::TAIL_BYTES`] past them.
///
/// The staged GEMV copies rows in aligned 16-byte vectors, the last of which
/// may run up to 15 bytes past a row, and the decoders read aligned words that
/// may run up to 3 bytes past a block; past a buffer's last block both stay
/// inside the allocation. Weights reach the GGML kernels only as these.
#[derive(Clone, Debug)]
pub struct GgmlBlocks {
    /// Exactly the blocks, a view of the longer allocation.
    blocks: DeviceBuffer<u8>,
}

impl GgmlBlocks {
    /// Bytes the allocation holds past the blocks: one aligned vector.
    pub const TAIL_BYTES: usize = 16;

    /// An uninitialized allocation for `bytes` of blocks on `stream`.
    pub fn allocate(pool: &MemoryPool, stream: &Stream, bytes: usize) -> Result<Self> {
        let padded =
            bytes.checked_add(Self::TAIL_BYTES).ok_or(crate::Error::InvalidTensorSize {
                name: "GGML blocks".into(),
                expected: usize::MAX,
                actual: bytes,
            })?;
        let allocation = pool.allocate::<u8>(stream, padded)?;
        Ok(Self { blocks: allocation.slice(0..bytes)? })
    }

    /// The blocks, exactly as long as they are.
    #[must_use]
    pub const fn view(&self) -> &DeviceBuffer<u8> {
        &self.blocks
    }

    pub const fn view_mut(&mut self) -> &mut DeviceBuffer<u8> {
        &mut self.blocks
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.blocks.len()
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }
}
