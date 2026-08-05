use runtime::backend::{DecodeSequence, SamplingLogits};

use super::graph::DecodeResources;
use crate::{CudaSharedRoutedModelSession, Error, Result};

impl DecodeResources {
    pub(super) fn finish(
        &mut self,
        sessions: &mut [&mut CudaSharedRoutedModelSession],
        sequences: &[DecodeSequence],
    ) -> Result<Option<Vec<u32>>> {
        let policies =
            sequences.iter().map(|sequence| sequence.sampling_logits).collect::<Vec<_>>();
        let device_sampling = policies.iter().copied().all(device_policy);
        if device_sampling {
            self.sampler.sample(&self.logits, &policies)?;
            self.backend
                .inner
                .stream
                .copy_to_host(self.sampler.selected(), &mut self.token_staging)?;
        }
        self.commit(sessions, !device_sampling)?;
        if device_sampling {
            self.token_staging.to_vec().map(Some).map_err(Into::into)
        } else {
            Ok(None)
        }
    }

    fn commit(
        &mut self,
        sessions: &mut [&mut CudaSharedRoutedModelSession],
        copy_logits: bool,
    ) -> Result<()> {
        for index in 0..self.layers.len() {
            self.layers[index].commit(sessions, index)?;
        }
        for (row, session) in sessions.iter_mut().enumerate() {
            if copy_logits {
                let start = super::graph::checked(row, session.logits.len())?;
                self.backend.inner.stream.copy_device_range(
                    &self.logits,
                    start..start + session.logits.len(),
                    &mut session.logits,
                    0,
                )?;
            }
            session.position = session
                .position
                .checked_add(1)
                .ok_or(Error::InvalidDecoderKernel("shared-routed session position overflow"))?;
        }
        Ok(())
    }
}

const fn device_policy(policy: SamplingLogits) -> bool {
    matches!(
        policy,
        SamplingLogits::None | SamplingLogits::SampleTopK { .. } | SamplingLogits::Sample { .. }
    )
}
