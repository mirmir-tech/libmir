use mircuda::{DeviceBuffer, bf16};

use super::CudaBackend;
use crate::Result;

impl CudaBackend {
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
