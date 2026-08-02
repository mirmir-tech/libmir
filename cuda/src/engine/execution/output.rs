use mircuda::{DeviceBuffer, bf16};
use runtime::backend::{DecodeOutput, LogitsTrace, SamplingLogits, TokenEvent};

use crate::{
    CudaBackend, CudaClampedRoutedModelSession, CudaMoeModelSession, CudaSharedRoutedModelSession,
    Result,
};

pub(in crate::engine) struct Output {
    pub(in crate::engine) token: Option<u32>,
    pub(in crate::engine) logits: Option<LogitsTrace>,
}

pub(in crate::engine) trait DeviceLogitsSession {
    fn sample(&mut self, policy: SamplingLogits) -> Result<&DeviceBuffer<u32>>;
    fn logits(&self) -> &DeviceBuffer<bf16>;
}

impl DeviceLogitsSession for CudaMoeModelSession {
    fn sample(&mut self, policy: SamplingLogits) -> Result<&DeviceBuffer<u32>> {
        Self::sample(self, policy)
    }

    fn logits(&self) -> &DeviceBuffer<bf16> {
        Self::logits(self)
    }
}

impl DeviceLogitsSession for CudaSharedRoutedModelSession {
    fn sample(&mut self, policy: SamplingLogits) -> Result<&DeviceBuffer<u32>> {
        Self::sample(self, policy)
    }

    fn logits(&self) -> &DeviceBuffer<bf16> {
        Self::logits(self)
    }
}

impl DeviceLogitsSession for CudaClampedRoutedModelSession {
    fn sample(&mut self, policy: SamplingLogits) -> Result<&DeviceBuffer<u32>> {
        Self::sample(self, policy)
    }

    fn logits(&self) -> &DeviceBuffer<bf16> {
        Self::logits(self)
    }
}

pub(in crate::engine) const fn device_sampling(policy: SamplingLogits) -> bool {
    matches!(
        policy,
        SamplingLogits::None | SamplingLogits::SampleTopK { .. } | SamplingLogits::Sample { .. }
    )
}

pub(in crate::engine) fn generation_output(
    backend: &CudaBackend,
    session: &mut impl DeviceLogitsSession,
    policy: SamplingLogits,
) -> Result<Output> {
    if device_sampling(policy) {
        let selected = session.sample(policy)?;
        return Ok(Output {
            token: Some(backend.read_token(selected)?),
            logits: None,
        });
    }
    output_logits(backend, session.logits())
}

pub(super) fn output_logits(backend: &CudaBackend, logits: &DeviceBuffer<bf16>) -> Result<Output> {
    let values = backend.read_logits(logits)?;
    Ok(Output {
        token: None,
        logits: Some(LogitsTrace {
            shape: vec![1, 1, i32::try_from(values.len())?],
            values,
        }),
    })
}

pub(in crate::engine) fn decode_output(output: Output) -> DecodeOutput {
    DecodeOutput {
        event: TokenEvent {
            token_id: output.token,
            text: "cuda.decode=device-token-pipeline".into(),
            finished: false,
        },
        logits: output.logits,
        candidates: None,
        timings: None,
    }
}
