use crate::{Error, Result};

#[derive(Clone, Copy)]
pub(super) struct Layout {
    pub(super) packed_per_matrix: usize,
    pub(super) groups_per_matrix: usize,
}

pub(super) fn product(left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right).ok_or(Error::InvalidQuantizedGemv("shape overflow"))
}

pub(super) fn indexed(per_matrix: usize, matrix_index: usize) -> Result<usize> {
    product(
        per_matrix,
        matrix_index
            .checked_add(1)
            .ok_or(Error::InvalidQuantizedGemv("matrix index overflow"))?,
    )
}

pub(super) fn narrow(value: usize) -> Result<u32> {
    Ok(u32::try_from(value)?)
}

pub(super) const fn require(operand: &'static str, expected: usize, actual: usize) -> Result<()> {
    if actual < expected {
        return Err(Error::QuantizedGemvLengthMismatch { operand, expected, actual });
    }
    Ok(())
}
