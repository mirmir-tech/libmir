use runtime::backend::SamplingLogits;

use crate::{
    config::DecodeReservation,
    engine::{Array, DecodeCapacity, DecoderModel, Stream, paged_attention_min_context},
    native::{error::Result, model::NativeOutput, session::SessionState, step},
};

/// Prefix insertion can share the last prompt page after prefill admission.
/// Fit the first device-token write before it starts, reclaiming only this
/// state's optional tail; the state is not yet in the live session registry.
pub(in crate::native::prefill) fn output(
    model: &DecoderModel,
    stream: &Stream,
    state: &mut SessionState,
    logits: Array,
    sampling: SamplingLogits,
) -> Result<NativeOutput> {
    if stream.config().cache.decode_reservation == DecodeReservation::GenerationBudget
        && step::supports_device_token(sampling.clone())
    {
        let check = |state: &SessionState| {
            state.cache.plan_decode_capacity(
                1,
                paged_attention_min_context(stream),
                &mut DecodeCapacity::default(),
            )
        };
        match check(state) {
            Err(crate::engine::Error::KvPageCapacity { .. }) => {
                let page = stream.config().kv_cache.block_size.max(1);
                state.cache.release_reservation_after(state.position.saturating_add(page))?;
                check(state)?;
            },
            result => result?,
        }
    }
    step::output(model, stream, state, logits, sampling)
}
