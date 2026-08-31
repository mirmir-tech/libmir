use mircuda::{DeviceBuffer, PinnedBuffer, bf16};

use crate::{CudaBackend, ExecutionPhase, Result, kernels::ActivationFingerprint};

pub(super) struct LayerFingerprintTrace {
    kernel: ActivationFingerprint,
    device: DeviceBuffer<u64>,
    host: PinnedBuffer<u64>,
    stream: mircuda::Stream,
    phase: ExecutionPhase,
    tokens: usize,
}

impl LayerFingerprintTrace {
    pub(super) fn new(
        backend: &CudaBackend,
        layers: usize,
        elements: usize,
        phase: ExecutionPhase,
        tokens: usize,
    ) -> Result<Self> {
        let values = layers * 4;
        Ok(Self {
            kernel: ActivationFingerprint::compile(&backend.inner.compiler, elements)?,
            device: backend.inner.pool.allocate(&backend.inner.stream, values)?,
            host: backend.inner.context.allocate_pinned(values)?,
            stream: backend.inner.stream.clone(),
            phase,
            tokens,
        })
    }

    pub(super) fn record(&mut self, input: &DeviceBuffer<bf16>, layer: usize) -> Result<()> {
        self.kernel.execute(&self.stream, input, &mut self.device, layer)
    }

    pub(super) fn publish(&mut self) -> Result<()> {
        self.stream.copy_to_host(&self.device, &mut self.host)?;
        let host = self.host.to_vec()?;
        for (checkpoint, values) in host.as_chunks::<2>().0.iter().enumerate() {
            tracing::info!(
                target: "libmir_cuda::layer_fingerprint",
                phase = ?self.phase,
                tokens = self.tokens,
                layer = checkpoint / 2,
                stage = if checkpoint.is_multiple_of(2) { "attention" } else { "output" },
                sum_bits = values[0],
                weighted = values[1],
                "activation fingerprint"
            );
        }
        Ok(())
    }
}
