use super::layer::HybridLinearMoeLayer;
use crate::{
    FusionMode,
    engine::{Result, Stream, expert_fusion::fits_additional_fusion},
};

pub(super) fn prepare(layers: &mut [HybridLinearMoeLayer], stream: &Stream) -> Result<()> {
    let mode = if stream.config().fusion.shared_dense_gate_up.enabled() {
        FusionMode::Enabled
    } else if stream.config().tuning.mode == runtime::tuning::TuningMode::Disabled {
        FusionMode::Disabled
    } else {
        FusionMode::Auto
    };
    let additional = additional_bytes(layers)?;
    let enabled = match (mode, additional) {
        (FusionMode::Enabled, Some(_)) => true,
        (FusionMode::Auto, Some(bytes)) => fits_additional_fusion(stream, bytes)?,
        (FusionMode::Disabled, _) | (_, None) => false,
    };
    if enabled {
        for layer in layers {
            let _enabled = layer.enable_decode_plan_candidate(stream)?;
        }
        stream.synchronize()?;
    }
    tracing::info!(
        target: "libmir::metal::tuning",
        ?mode,
        enabled,
        additional_bytes = ?additional,
        "prepared complete Metal decode plan candidates"
    );
    Ok(())
}

fn additional_bytes(layers: &[HybridLinearMoeLayer]) -> Result<Option<usize>> {
    let mut total = 0_usize;
    for layer in layers {
        let Some(bytes) = layer.decode_plan_candidate_bytes()? else {
            return Ok(None);
        };
        total = total.checked_add(bytes).ok_or(crate::engine::Error::ShapeOverflow)?;
    }
    Ok(Some(total))
}
