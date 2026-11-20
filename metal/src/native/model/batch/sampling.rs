use runtime::backend::SamplingLogits;
use uuid::Uuid;

use super::{DecodeInput, SessionState};
use crate::{
    engine::Array,
    native::{
        error::{Error, Result},
        step,
    },
};

pub(super) fn sample_batch(
    inputs: &[DecodeInput],
    states: &mut [(Uuid, SessionState)],
    stream: &crate::engine::Stream,
) -> Result<Array> {
    let logits = inputs
        .iter()
        .zip(states.iter_mut())
        .map(|(input, (_, state))| step::take_pending(state, input.token))
        .collect::<Result<Vec<_>>>()?;
    let refs = logits.iter().collect::<Vec<_>>();
    // Greedy and shared-parameter rows sample in one graph; mixed rows one by one.
    if inputs.iter().all(|input| input.sampling == SamplingLogits::None) {
        return Ok(Array::concatenate(&refs, 0, stream)?.argmax(stream)?);
    }
    if let Some((parameters, draws)) =
        step::shared_device_sampling(inputs.iter().map(|input| &input.sampling))
    {
        let rows = Array::concatenate(&refs, 0, stream)?;
        return Ok(crate::engine::sample_rows(&rows, parameters, &draws, stream)?);
    }
    let sampled = inputs
        .iter()
        .zip(&logits)
        .map(|(input, logits)| {
            step::device_token(logits, &input.sampling, stream)?.ok_or_else(|| {
                Error::InvalidDecodeBatch("packed row does not use device sampling".into())
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let refs = sampled.iter().collect::<Vec<_>>();
    Ok(Array::concatenate(&refs, 0, stream)?)
}
