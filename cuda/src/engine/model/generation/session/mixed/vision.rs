//! Spatial-merge vision prefill of a new session.

use super::{super::super::super::SpatialVisionPrefill, MixedMixerExecution};
use crate::{
    CudaBackend, Result,
    engine::execution::{Output, generation_output},
};

impl MixedMixerExecution {
    pub(super) fn prefill_vision_session(
        &mut self,
        backend: &CudaBackend,
        input: SpatialVisionPrefill<'_>,
    ) -> Result<Output> {
        let mut session = self.template.instantiate_with_caches(&self.caches)?;
        session.prefill_vision(
            input.session,
            input.tokens,
            input.positions,
            input.table,
            input.image_span,
            input.image,
            input.position_delta,
        )?;
        let output = generation_output(backend, &mut session, input.sampling)?;
        self.sessions.insert(input.session, session);
        Ok(output)
    }
}
