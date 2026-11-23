//! Output-row selection on an affine projection: each row carries its own
//! packed values, scales and biases, so rows reorder without dequantizing.

use super::QuantizedLinear;
use crate::engine::{Array, QuantizedArrays, Result, Stream};

impl QuantizedLinear {
    /// The projection whose output row `i` is this one's row `rows[i]`.
    pub(in crate::engine) fn take_output_rows(
        &self,
        rows: &Array,
        stream: &Stream,
    ) -> Result<Self> {
        let take = |array: &Array| array.take(rows, 0, stream);
        let arrays = QuantizedArrays::new(
            take(&self.arrays.weight)?,
            take(&self.arrays.scales)?,
            take(&self.arrays.biases)?,
            self.group_size,
            self.bits,
        )?;
        Ok(Self {
            arrays,
            bias: self.bias.as_ref().map(take).transpose()?,
            group_size: self.group_size,
            bits: self.bits,
        })
    }
}
