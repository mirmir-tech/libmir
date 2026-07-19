use std::path::Path;

use super::{Array, Dtype, Result, Stream, array::native_shape};

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

    pub fn matmul(&self, right: &Self, stream: &Stream) -> Result<Self> {
        Self::from_native(stream.native().graph().matmul(self.native(), right.native())?)
    }

    pub fn layer_norm(
        &self,
        weight: &Self,
        bias: &Self,
        eps: f32,
        stream: &Stream,
    ) -> Result<Self> {
        Self::from_native(stream.native().graph().layer_norm(
            self.native(),
            weight.native(),
            bias.native(),
            eps,
        )?)
    }

    pub fn gelu_tanh(&self, stream: &Stream) -> Result<Self> {
        let graph = stream.native().graph();
        let output = graph.gelu_tanh(self.native())?;
        Self::from_native(graph.astype(&output, self.native().dtype()?)?)
    }

    pub fn gelu(&self, stream: &Stream) -> Result<Self> {
        let graph = stream.native().graph();
        let output = graph.gelu(self.native())?;
        Self::from_native(graph.astype(&output, self.native().dtype()?)?)
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

    pub fn astype(&self, dtype: Dtype, stream: &Stream) -> Result<Self> {
        Self::from_native(stream.native().graph().astype(self.native(), dtype.native()?)?)
    }

    pub fn cos(&self, stream: &Stream) -> Result<Self> {
        Self::from_native(stream.native().graph().cos(self.native())?)
    }

    pub fn sin(&self, stream: &Stream) -> Result<Self> {
        Self::from_native(stream.native().graph().sin(self.native())?)
    }

    pub fn reduce_sum(&self, axis: i32, keepdims: bool, stream: &Stream) -> Result<Self> {
        Self::from_native(stream.native().graph().reduce_sum(self.native(), axis, keepdims)?)
    }

    pub fn clip(&self, minimum: &Self, maximum: &Self, stream: &Stream) -> Result<Self> {
        Self::from_native(stream.native().graph().clip(
            self.native(),
            minimum.native(),
            maximum.native(),
        )?)
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
