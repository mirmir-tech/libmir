use mircuda::{Context, DeviceBuffer, Stream, bf16};

use super::candidate::Candidate;
use crate::{DensePlanRequest, Error, Result};

const ABSOLUTE_TOLERANCE: f32 = 0.125;
const RELATIVE_TOLERANCE: f32 = 0.01;

#[allow(clippy::too_many_arguments)]
pub(super) fn retain_compatible(
    context: &Context,
    stream: &Stream,
    request: DensePlanRequest,
    candidates: &mut Vec<Candidate>,
    fallback: usize,
    input: &DeviceBuffer<bf16>,
    weight: &DeviceBuffer<bf16>,
    output: &mut DeviceBuffer<bf16>,
) -> Result<usize> {
    candidates
        .get_mut(fallback)
        .ok_or(Error::InvalidExecutionPlan("dense tuner fallback is missing"))?
        .plan
        .execute(stream, input, weight, output)?;
    let reference = read(context, stream, output)?;
    let mut compatible = Vec::with_capacity(candidates.len());
    for (index, candidate) in candidates.iter_mut().enumerate() {
        let accepted = if index == fallback {
            true
        } else {
            candidate.plan.execute(stream, input, weight, output)?;
            equivalent(&reference, &read(context, stream, output)?)
        };
        if !accepted {
            tracing::warn!(
                ?request,
                execution = ?candidate.execution,
                "rejected numerically incompatible CUDA dense tuning candidate"
            );
        }
        compatible.push(accepted);
    }
    let mut retained = Vec::with_capacity(candidates.len());
    let mut retained_fallback = None;
    for (index, candidate) in std::mem::take(candidates).into_iter().enumerate() {
        if compatible[index] {
            if index == fallback {
                retained_fallback = Some(retained.len());
            }
            retained.push(candidate);
        }
    }
    *candidates = retained;
    retained_fallback.ok_or(Error::InvalidExecutionPlan("dense tuner rejected its fallback"))
}

fn read(context: &Context, stream: &Stream, output: &DeviceBuffer<bf16>) -> Result<Vec<bf16>> {
    let mut host = context.allocate_pinned::<bf16>(output.len())?;
    stream.copy_to_host(output, &mut host)?;
    Ok(host.to_vec()?)
}

fn equivalent(reference: &[bf16], candidate: &[bf16]) -> bool {
    reference.len() == candidate.len()
        && reference.iter().zip(candidate).all(|(reference, candidate)| {
            let reference = reference.to_f32();
            let candidate = candidate.to_f32();
            reference.is_finite()
                && candidate.is_finite()
                && (reference - candidate).abs()
                    <= ABSOLUTE_TOLERANCE.max(reference.abs() * RELATIVE_TOLERANCE)
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_bf16_rounding_and_rejects_material_drift() {
        let reference = [1.0, -20.0, 0.0].map(bf16::from_f32);
        let close = [1.125, -20.125, 0.125].map(bf16::from_f32);
        let drift = [1.5, -20.0, 0.0].map(bf16::from_f32);

        assert!(equivalent(&reference, &close));
        assert!(!equivalent(&reference, &drift));
    }
}
