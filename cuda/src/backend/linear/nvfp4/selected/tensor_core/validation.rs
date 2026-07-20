use crate::{Error, Result};

pub(super) const fn require_len(
    operand: &'static str,
    expected: usize,
    actual: usize,
) -> Result<()> {
    if actual < expected {
        Err(Error::QuantizedGemvLengthMismatch { operand, expected, actual })
    } else {
        Ok(())
    }
}
