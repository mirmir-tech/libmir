use runtime::backend::{DecodeOutput, DecodeSequence};

use super::Model;
use crate::{Result, scheduler::PendingModelDecode};

impl Model {
    pub(crate) fn start_decode_sequence(
        &self,
        sequence: DecodeSequence,
    ) -> Result<PendingModelDecode> {
        self.inner.coordinator.start_decode(sequence)
    }

    pub(crate) fn finish_decode_sequence(
        &self,
        pending: PendingModelDecode,
    ) -> Result<DecodeOutput> {
        self.inner.coordinator.finish_decode(pending)
    }
}
