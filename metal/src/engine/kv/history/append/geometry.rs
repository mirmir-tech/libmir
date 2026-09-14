use crate::engine::{Error, Result};

pub(super) struct Geometry {
    pub constants: [u32; 4],
    pub grid: [usize; 3],
    pub group: [usize; 3],
}

impl Geometry {
    pub fn new(shape: [usize; 4], offset: usize) -> Result<Self> {
        let [batch, heads, capacity, dim] = shape;
        if shape.contains(&0) || offset >= capacity {
            return Err(Error::InvalidModel("history append is outside its allocation".into()));
        }
        // The Metal kernel addresses the entire allocation using uint indices.
        let total = shape
            .into_iter()
            .try_fold(1_usize, usize::checked_mul)
            .ok_or(Error::ShapeOverflow)?;
        let _address_limit = u32::try_from(total)?;
        let elements = batch * heads * dim;
        Ok(Self {
            constants: [
                u32::try_from(heads)?,
                u32::try_from(dim)?,
                u32::try_from(capacity)?,
                u32::try_from(offset)?,
            ],
            grid: [elements, 1, 1],
            group: [elements.min(256), 1, 1],
        })
    }

    pub fn dispatch(&self) -> mirtal::AliasingDispatch {
        mirtal::AliasingDispatch::new([0, 1])
            .constants(self.constants)
            .strides(
                [2, 3]
                    .into_iter()
                    .flat_map(|input| [0, 1, 3].map(|axis| mirtal::StrideBinding { input, axis })),
            )
            .grid(self.grid)
            .threadgroup(self.group)
    }
}
