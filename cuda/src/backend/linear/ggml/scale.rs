//! A factor folded into a GGML projection's output scale.

use super::GgmlCheckpointWeight;
use crate::{CudaBackend, Error, Result};

impl GgmlCheckpointWeight {
    /// The same projection with its products times `factor`, folded into its
    /// output scale (one per matrix).
    pub(crate) fn scaled(&self, backend: &CudaBackend, factor: f32) -> Result<Self> {
        // The output scale multiplies the product before the bias is added,
        // so a factor on the whole projection would leave the bias unscaled.
        if self.bias.is_some() {
            return Err(Error::InvalidExecutionPlan("a biased GGML projection takes no factor"));
        }
        let inner = &backend.inner;
        let mut values = vec![factor; self.matrices];
        if let Some(scale) = &self.scale {
            let mut host = inner.context.allocate_pinned::<f32>(scale.len())?;
            inner.stream.copy_to_host(scale, &mut host)?;
            for (value, existing) in values.iter_mut().zip(host.to_vec()?) {
                *value *= existing;
            }
        }
        let mut host = inner.context.allocate_pinned::<f32>(values.len())?;
        host.copy_from_slice(&values)?;
        let mut scale = inner.pool.allocate::<f32>(&inner.stream, values.len())?;
        inner.stream.copy_to_device(&mut host, &mut scale)?;
        inner.stream.synchronize()?;
        Ok(Self { scale: Some(scale), ..self.clone() })
    }
}
