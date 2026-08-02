use mircuda::{Compiler, Context, DeviceBuffer, MemoryPool, Stream, bf16};

use super::{CudaBackend, tuning::CudaAutoTuner};
use crate::Result;

impl CudaBackend {
    pub(crate) fn compiler(&self) -> &Compiler {
        &self.inner.compiler
    }

    pub(crate) fn context(&self) -> &Context {
        &self.inner.context
    }

    pub(crate) fn stream(&self) -> &Stream {
        &self.inner.stream
    }

    pub(crate) fn pool(&self) -> &MemoryPool {
        &self.inner.pool
    }

    pub(crate) fn auto_tuner(&self) -> &CudaAutoTuner {
        &self.inner.tuner
    }

    pub(crate) fn finish_startup_tuning(&self) {
        self.inner.tuner.finish_startup();
    }

    #[cfg(all(test, target_os = "linux"))]
    pub(crate) fn test_resources(&self) -> (&Context, &Stream, &MemoryPool, &Compiler) {
        (&self.inner.context, &self.inner.stream, &self.inner.pool, &self.inner.compiler)
    }

    pub(crate) fn read_token(&self, token: &DeviceBuffer<u32>) -> Result<u32> {
        let mut host = self.inner.context.allocate_pinned::<u32>(1)?;
        self.inner.stream.copy_to_host(token, &mut host)?;
        Ok(host.to_vec()?[0])
    }

    pub(crate) fn read_tokens(&self, tokens: &DeviceBuffer<u32>) -> Result<Vec<u32>> {
        let mut host = self.inner.context.allocate_pinned::<u32>(tokens.len())?;
        self.inner.stream.copy_to_host(tokens, &mut host)?;
        Ok(host.to_vec()?)
    }

    pub(crate) fn read_logits(&self, logits: &DeviceBuffer<bf16>) -> Result<Vec<f32>> {
        let mut host = self.inner.context.allocate_pinned::<bf16>(logits.len())?;
        self.inner.stream.copy_to_host(logits, &mut host)?;
        Ok(host.to_vec()?.into_iter().map(bf16::to_f32).collect())
    }
}
