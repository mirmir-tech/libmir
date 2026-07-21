use super::{Array, Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    Bool,
    Uint8,
    Uint32,
    Int32,
    Float16,
    Bfloat16,
    Float32,
    Unknown,
}

impl Array {
    pub fn shape(&self) -> Result<Vec<i32>> {
        let shape = self
            .native()
            .shape()?
            .dimensions()
            .iter()
            .copied()
            .map(i32::try_from)
            .collect::<std::result::Result<_, _>>()?;
        Ok(shape)
    }

    pub fn dtype(&self) -> Result<Dtype> {
        Ok(match self.native().dtype()? {
            mirtal::DType::Bool => Dtype::Bool,
            mirtal::DType::Uint8 => Dtype::Uint8,
            mirtal::DType::Uint32 => Dtype::Uint32,
            mirtal::DType::Int32 => Dtype::Int32,
            mirtal::DType::Float16 => Dtype::Float16,
            mirtal::DType::Bfloat16 => Dtype::Bfloat16,
            mirtal::DType::Float32 => Dtype::Float32,
        })
    }

    pub fn byte_len(&self) -> Result<usize> {
        let elements = self.shape()?.into_iter().try_fold(1_usize, |total, dimension| {
            let dimension = usize::try_from(dimension)?;
            total.checked_mul(dimension).ok_or(Error::ShapeOverflow)
        })?;
        let bytes_per_element = match self.dtype()? {
            Dtype::Bool | Dtype::Uint8 => 1,
            Dtype::Uint32 | Dtype::Int32 | Dtype::Float32 => 4,
            Dtype::Float16 | Dtype::Bfloat16 => 2,
            Dtype::Unknown => return Err(Error::InvalidModel("unknown MLX dtype".into())),
        };
        elements.checked_mul(bytes_per_element).ok_or(Error::ShapeOverflow)
    }
}

impl Dtype {
    pub(super) fn native(self) -> Result<mirtal::DType> {
        match self {
            Self::Bool => Ok(mirtal::DType::Bool),
            Self::Uint8 => Ok(mirtal::DType::Uint8),
            Self::Uint32 => Ok(mirtal::DType::Uint32),
            Self::Int32 => Ok(mirtal::DType::Int32),
            Self::Float16 => Ok(mirtal::DType::Float16),
            Self::Bfloat16 => Ok(mirtal::DType::Bfloat16),
            Self::Float32 => Ok(mirtal::DType::Float32),
            Self::Unknown => Err(Error::InvalidModel("unknown MLX dtype".into())),
        }
    }
}
