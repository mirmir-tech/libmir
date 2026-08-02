use foundation::model::BackendTarget;
use runtime::{
    backend::{PrefillOutput, PrefillRequest},
    progress::ProgressEvent,
};

use super::Model;
use crate::Result;

impl Model {
    pub(crate) fn prefill_request(
        &self,
        request: PrefillRequest,
        expects_decode: bool,
        progress: &mut dyn FnMut(ProgressEvent),
    ) -> Result<PrefillOutput> {
        if matches!(self.inner.engine.target(), BackendTarget::Cuda | BackendTarget::Metal) {
            self.inner.coordinator.submit_prefill(request, expects_decode, progress)
        } else {
            Ok(self.inner.engine.prefill_request_with_progress(&request, progress)?)
        }
    }
}
