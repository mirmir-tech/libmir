//! llama.cpp's tiled value heads. When a Gated `DeltaNet` layer has more
//! value than key heads, the GGUF converter stores value head `j` of key head
//! `k` at `j * key_heads + k` (`[K0, K1, .., K0, K1, ..]`); the Hugging Face
//! layout the kernels read groups them as `k * per_key + j`.

use crate::error::{ModelsError, Result};

/// Value heads stored tiled across key heads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TiledValueHeads {
    key_heads: usize,
    per_key: usize,
}

impl TiledValueHeads {
    /// The tiling of `value_heads` over `key_heads`; `None` when value heads
    /// do not outnumber key heads, so stored and grouped order agree.
    pub fn new(key_heads: usize, value_heads: usize) -> Result<Option<Self>> {
        if key_heads == 0 || !value_heads.is_multiple_of(key_heads) {
            return Err(ModelsError::InvalidConfig(format!(
                "{value_heads} value heads do not group over {key_heads} key heads"
            )));
        }
        Ok((value_heads > key_heads).then(|| Self {
            key_heads,
            per_key: value_heads / key_heads,
        }))
    }

    #[must_use]
    pub const fn key_heads(self) -> usize {
        self.key_heads
    }

    /// Value heads per key head.
    #[must_use]
    pub const fn per_key(self) -> usize {
        self.per_key
    }

    /// The stored head of grouped head `grouped`.
    #[must_use]
    pub const fn stored_head(self, grouped: usize) -> usize {
        (grouped % self.per_key) * self.key_heads + grouped / self.per_key
    }

    /// The stored row of every grouped row: `offset` leading rows in place,
    /// then `width` rows per value head in grouped order.
    pub fn stored_rows(self, offset: usize, width: usize) -> Result<Vec<u32>> {
        let heads = self.key_heads * self.per_key;
        let rows = (0..offset).chain((0..heads).flat_map(|grouped| {
            let start = offset + self.stored_head(grouped) * width;
            start..start + width
        }));
        Ok(rows.map(u32::try_from).collect::<std::result::Result<_, _>>()?)
    }
}

#[cfg(test)]
mod tests {
    use super::TiledValueHeads;
    use crate::error::{ModelsError, Result};

    /// Two key heads with two value heads each: stored `[K0, K1, K0, K1]`.
    #[test]
    fn grouped_rows_read_their_tiled_heads() -> Result<()> {
        let tiled = TiledValueHeads::new(2, 4)?
            .ok_or_else(|| ModelsError::InvalidConfig("four value heads tile over two".into()))?;
        assert_eq!(tiled.stored_rows(1, 2)?, vec![0, 1, 2, 5, 6, 3, 4, 7, 8]);
        assert_eq!(TiledValueHeads::new(4, 4)?, None);
        assert!(TiledValueHeads::new(3, 4).is_err());
        Ok(())
    }
}
