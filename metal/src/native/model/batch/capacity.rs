use super::{DecodeInput, LoadedModel, NativeOutput};
use crate::{
    engine::{DecodeCapacity, paged_attention_min_context},
    native::{error::Result, step},
};

impl LoadedModel {
    pub fn decode(
        &mut self,
        session: uuid::Uuid,
        token: u32,
        sampling: runtime::backend::SamplingLogits,
    ) -> Result<NativeOutput> {
        let inputs = [DecodeInput { session, token, sampling }];
        self.validate_decode_inputs(&inputs)?;
        self.validate_decode_capacity(&inputs)?;
        let result = self.decode_admitted(session, token, sampling);
        self.finish_decode(&inputs, result)
    }

    pub(super) fn validate_decode_capacity(&mut self, inputs: &[DecodeInput]) -> Result<()> {
        let checked = self.check_decode_capacity(inputs);
        if matches!(
            &checked,
            Err(crate::native::error::Error::Engine(crate::engine::Error::KvPageCapacity { .. }))
        ) && self.stream.config().cache.decode_reservation
            == crate::config::DecodeReservation::GenerationBudget
        {
            self.reclaim_decode_reservations()?;
            return self.check_decode_capacity(inputs);
        }
        checked
    }

    fn check_decode_capacity(&self, inputs: &[DecodeInput]) -> Result<()> {
        let threshold = paged_attention_min_context(&self.stream);
        let mut plan = DecodeCapacity::default();
        for input in inputs {
            let state = self.session(input.session)?;
            let forwards = usize::from(state.pending.is_none())
                + usize::from(step::supports_device_token(input.sampling));
            state.cache.plan_decode_capacity(forwards, threshold, &mut plan)?;
        }
        Ok(())
    }
}
