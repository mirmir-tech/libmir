//! Greedy verification: the decoder's argmax after every row of a pass, and
//! the single-row inputs of a further draft.

use mircuda::{DeviceBuffer, bf16};
use runtime::backend::SamplingLogits;

use super::PredictionExecution;
use crate::{
    DeviceBatchSamplerBf16, Error, Result,
    backend::shared::routed::{CudaSharedRoutedModelTemplate, boundary::SharedRoutedOutputHead},
};

/// The output head and sampler over every row, built on first use: only the
/// short verification shapes need them.
#[derive(Debug)]
pub(super) struct Verification {
    head: SharedRoutedOutputHead,
    logits: DeviceBuffer<bf16>,
    sampler: DeviceBatchSamplerBf16,
    policies: Vec<SamplingLogits>,
}

impl PredictionExecution {
    /// The decoder's greedy token after every normalized row, read to the
    /// host.
    pub(in crate::backend::shared::routed) fn decoder_argmax(
        &mut self,
        template: &CudaSharedRoutedModelTemplate,
    ) -> Result<Vec<u32>> {
        let verification = self.verification(template)?;
        let selected = verification.sampler.sample(&verification.logits, &verification.policies)?;
        template.backend.read_tokens(selected)
    }

    /// The decoder's logits after every normalized row, read to the host.
    pub(in crate::backend::shared::routed) fn decoder_logits(
        &mut self,
        template: &CudaSharedRoutedModelTemplate,
    ) -> Result<Vec<f32>> {
        let verification = self.verification(template)?;
        template.backend.read_logits(&verification.logits)
    }

    /// The output head over every row, computed into the verification
    /// logits.
    fn verification(
        &mut self,
        template: &CudaSharedRoutedModelTemplate,
    ) -> Result<&mut Verification> {
        let backend = &template.backend;
        if self.verification.is_none() {
            let vocab = template.decoder.vocab_size;
            let logits = self
                .tokens
                .checked_mul(vocab)
                .ok_or(Error::InvalidDecoderKernel("verification logits size overflow"))?;
            self.verification = Some(Verification {
                head: template.prepare_output_head(self.tokens)?,
                logits: backend.inner.pool.allocate(&backend.inner.stream, logits)?,
                sampler: backend.prepare_device_batch_sampler_bf16(vocab, self.tokens)?,
                policies: vec![SamplingLogits::None; self.tokens],
            });
        }
        let verification = self
            .verification
            .as_mut()
            .ok_or(Error::InvalidDecoderKernel("verification head is missing"))?;
        verification.head.execute(&self.hidden, &mut verification.logits)?;
        Ok(verification)
    }

    /// Loads a single-row draft input from the device: the module's last
    /// output and the token drafted from it.
    pub(in crate::backend::shared::routed) fn load_row(
        &mut self,
        hidden: &DeviceBuffer<bf16>,
        token: &DeviceBuffer<u32>,
    ) -> Result<()> {
        if self.tokens != 1 || token.len() != 1 || hidden.len() != self.hidden.len() {
            return Err(Error::InvalidDecoderKernel("draft row input differs from plan"));
        }
        let stream = &self.backend.inner.stream;
        stream.copy_device_range(hidden, 0..hidden.len(), &mut self.hidden, 0)?;
        stream.copy_device_range(token, 0..1, &mut self.token_ids, 0)?;
        Ok(())
    }
}
