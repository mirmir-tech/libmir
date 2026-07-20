use runtime::kv::KvCacheDType;

use crate::engine::{Error, Result};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum KvPageFormat {
    #[default]
    Native,
    Int8PerTokenHead,
}

impl KvPageFormat {
    pub(crate) fn resolve(dtype: KvCacheDType) -> Result<Self> {
        match dtype {
            KvCacheDType::Auto | KvCacheDType::BFloat16 => Ok(Self::Native),
            KvCacheDType::Int8PerTokenHead => Ok(Self::Int8PerTokenHead),
            unsupported => Err(Error::InvalidModel(format!(
                "Metal K/V cache dtype `{unsupported}` is not implemented; use auto, bfloat16, or int8_per_token_head"
            ))),
        }
    }

    pub(crate) const fn quantized(self) -> bool {
        matches!(self, Self::Int8PerTokenHead)
    }

    pub(crate) fn packed_words(self, head_dim: usize) -> Result<usize> {
        match self {
            Self::Native => Ok(head_dim),
            Self::Int8PerTokenHead => {
                Ok(mirtal::SymmetricQuantization::new(8)?.packed_words(head_dim))
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use runtime::kv::KvCacheDType;

    use super::KvPageFormat;
    use crate::engine::Result;

    #[test]
    fn resolves_only_truthful_metal_formats() -> Result<()> {
        assert_eq!(KvPageFormat::resolve(KvCacheDType::Auto)?, KvPageFormat::Native);
        assert_eq!(
            KvPageFormat::resolve(KvCacheDType::Int8PerTokenHead)?,
            KvPageFormat::Int8PerTokenHead
        );
        assert!(KvPageFormat::resolve(KvCacheDType::Int4PerTokenHead).is_err());
        Ok(())
    }
}
