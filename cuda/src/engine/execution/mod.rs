use std::time::Instant;

use runtime::backend::{DecodeOutput, DecodeRequest};

use super::{
    CudaEngine,
    model::{DeviceToken, ModelExecution, ModelRunner},
    profile::DecodeProfile,
};
use crate::{Error, Result};

mod output;
mod prefill;
mod step;

pub(super) use output::{Output, decode_output, device_sampling, generation_output};
pub use prefill::CudaPrefillBatch;
pub use step::CudaGenerationStepOutput;

impl CudaEngine {
    pub fn decode_token(&self, request: &DecodeRequest) -> Result<DecodeOutput> {
        let loaded = self.model(&request.model.id)?;
        let waiting = Instant::now();
        let mut runner = loaded.decode_runner()?;
        let wait = waiting.elapsed();
        loaded.require_session(request.session_id)?;
        let profile = DecodeProfile::begin(&self.backend, wait, 1, self.profile_decode())?;
        let mut output = self.decode_with_runner(&mut runner, request)?;
        drop(runner);
        if let Some(profile) = profile {
            profile.finish(&self.backend, std::slice::from_mut(&mut output))?;
        }
        Ok(output)
    }

    pub(super) fn decode_with_runner(
        &self,
        runner: &mut ModelRunner,
        request: &DecodeRequest,
    ) -> Result<DecodeOutput> {
        let selected = DeviceToken {
            session: request.session_id,
            token: request.token_id,
        };
        let use_device_token = runner.selected == Some(selected);
        let ModelExecution::Generation(generation) = &mut runner.execution else {
            return Err(Error::State("CUDA task is not a generation runner".into()));
        };
        let output = generation.decode(&self.backend, request, use_device_token)?;
        runner.selected =
            output.token.map(|token| DeviceToken { session: request.session_id, token });
        Ok(decode_output(output))
    }
}
