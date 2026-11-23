//! An affine projection's weight as a dense matrix, for layers that fold
//! weights into other products at load.

use super::QuantizedLinear;
use crate::engine::{Array, Error, Result, Stream};

impl QuantizedLinear {
    /// The dequantized `[output, input]` weight; a biased projection has no
    /// dense form.
    pub(in crate::engine) fn dense(&self, stream: &Stream) -> Result<Array> {
        if self.bias.is_some() {
            return Err(Error::InvalidQuantization(
                "a biased affine projection has no dense form".into(),
            ));
        }
        Array::from_native(stream.native().graph().dequantize(self.arrays.native())?)
    }
}
