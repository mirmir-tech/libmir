mod experts;
mod routing;

pub use experts::SortedExpertInputs;
pub use routing::RouterOutput;
pub(in crate::engine) use routing::route_top_k;

use super::{Array, Result, Stream};

impl Array {
    pub fn expand_dims(&self, axes: &[i32], stream: &Stream) -> Result<Self> {
        Self::from_native(stream.native().graph().expand_dims(self.native(), axes)?)
    }

    pub fn squeeze_axis(&self, axis: i32, stream: &Stream) -> Result<Self> {
        Self::from_native(stream.native().graph().squeeze_axis(self.native(), axis)?)
    }

    pub fn gelu_approx_mul(&self, input: &Self, stream: &Stream) -> Result<Self> {
        Self::from_native(stream.gelu_approx_mul(self.native(), input.native())?)?
            .astype_like(self, stream)
    }

    pub fn silu_mul(&self, input: &Self, stream: &Stream) -> Result<Self> {
        Self::from_native(stream.silu_mul(self.native(), input.native())?)?
            .astype_like(self, stream)
    }

    pub fn sigmoid_mul(&self, input: &Self, stream: &Stream) -> Result<Self> {
        let graph = stream.native().graph();
        let output = graph.sigmoid_multiply(self.native(), input.native())?;
        Self::from_native(graph.astype(&output, input.native().dtype()?)?)
    }
}
