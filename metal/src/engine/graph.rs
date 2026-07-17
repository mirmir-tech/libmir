use std::path::Path;

use super::{Array, Result, Stream, array::native_shape};

impl Array {
    pub fn export_graph_dot(&self, path: &Path) -> Result<()> {
        Ok(self.native().export_graph_dot(path)?)
    }

    pub fn rms_norm(&self, weight: &Self, eps: f32, stream: &Stream) -> Result<Self> {
        let graph = stream.native().graph();
        let output = graph.rms_norm(self.native(), weight.native(), eps)?;
        Self::from_native(graph.astype(&output, self.native().dtype()?)?)
    }

    pub fn multiply(&self, right: &Self, stream: &Stream) -> Result<Self> {
        Self::from_native(stream.native().graph().multiply(self.native(), right.native())?)
    }

    pub fn multiply_scalar(&self, scalar: f32, stream: &Stream) -> Result<Self> {
        let graph = stream.native().graph();
        let output = graph.multiply_scalar(self.native(), scalar)?;
        Self::from_native(graph.astype(&output, self.native().dtype()?)?)
    }

    pub fn logit_softcap(&self, cap: f32, stream: &Stream) -> Result<Self> {
        Self::from_native(stream.logit_softcap(self.native(), cap)?)?.astype_like(self, stream)
    }

    pub fn astype_like(&self, reference: &Self, stream: &Stream) -> Result<Self> {
        Self::from_native(
            stream.native().graph().astype(self.native(), reference.native().dtype()?)?,
        )
    }

    pub fn reshape(&self, shape: &[i32], stream: &Stream) -> Result<Self> {
        Self::from_native(stream.native().graph().reshape(self.native(), &native_shape(shape)?)?)
    }

    pub fn transpose(&self, axes: &[i32], stream: &Stream) -> Result<Self> {
        let axes = axes
            .iter()
            .copied()
            .map(usize::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Self::from_native(stream.native().graph().transpose(self.native(), &axes)?)
    }
}
