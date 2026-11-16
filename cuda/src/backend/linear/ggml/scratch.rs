use std::sync::{Arc, Mutex, Weak};

use mircuda::{DeviceBuffer, bf16};

use crate::{CudaBackend, Error, Result, kernels::GgmlActivations};

impl CudaBackend {
    /// BF16 scratch for one dequantized GGML matrix, shared by every prefill
    /// projection of the same size: they run in order on the backend stream.
    pub(in crate::backend) fn ggml_dequantize_scratch(
        &self,
        elements: usize,
    ) -> Result<Arc<Mutex<DeviceBuffer<bf16>>>> {
        let mut cache = self
            .inner
            .ggml_scratch
            .lock()
            .map_err(|_| Error::InvalidExecutionPlan("GGML scratch cache lock is poisoned"))?;
        if let Some(scratch) = cache.get(&elements).and_then(Weak::upgrade) {
            return Ok(scratch);
        }
        let buffer = self.inner.pool.allocate::<bf16>(&self.inner.stream, elements)?;
        let scratch = Arc::new(Mutex::new(buffer));
        cache.insert(elements, Arc::downgrade(&scratch));
        drop(cache);
        Ok(scratch)
    }

    /// Quantized decode activations shared by every GEMV plan of the same
    /// batch and input width, so projections of one input quantize it once.
    pub(super) fn ggml_activations(
        &self,
        tokens: usize,
        columns: usize,
    ) -> Result<Arc<Mutex<GgmlActivations>>> {
        let mut cache =
            self.inner.ggml_activations.lock().map_err(|_| {
                Error::InvalidExecutionPlan("GGML activation cache lock is poisoned")
            })?;
        if let Some(activations) = cache.get(&(tokens, columns)).and_then(Weak::upgrade) {
            return Ok(activations);
        }
        let activations = Arc::new(Mutex::new(GgmlActivations::new(
            &self.inner.pool,
            &self.inner.stream,
            tokens,
            columns,
        )?));
        cache.insert((tokens, columns), Arc::downgrade(&activations));
        drop(cache);
        Ok(activations)
    }
}
