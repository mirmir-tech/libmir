//! The gated activation a decoder configuration names.

use crate::{
    layout::{DecoderConfig, HiddenActivation},
    semantic::ActivationSpec,
};

/// The gated activation of `hidden_act`: aliases were resolved at config parse
/// time, and a missing name defaults to `SiLU` as Llama's configuration does.
pub(super) fn activation(decoder: &DecoderConfig) -> ActivationSpec {
    match decoder.hidden_activation.as_ref() {
        Some(HiddenActivation::GeluTanh) => ActivationSpec::GeluTanh,
        // gpt-oss fixes its clamped gate's sigmoid slope; only the limit is
        // configured.
        Some(HiddenActivation::Silu) | None => {
            decoder.swiglu_limit.map_or(ActivationSpec::SwiGlu, |limit| {
                ActivationSpec::ClampedSwiGlu { alpha: 1.702, limit }
            })
        },
        Some(other) => ActivationSpec::NamedGated { name: other.as_str().to_owned() },
    }
}
