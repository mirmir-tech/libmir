//! The gated activation a decoder configuration names.

use crate::{layout::DecoderConfig, semantic::ActivationSpec};

/// The gated activation of `hidden_act`, its aliases resolved once here:
/// transformers' `gelu_new` is the tanh approximation, `swish` is `SiLU`, and
/// a missing name defaults to `SiLU` as Llama's configuration does.
pub(super) fn activation(decoder: &DecoderConfig) -> ActivationSpec {
    match decoder.hidden_activation.as_deref() {
        Some("gelu_pytorch_tanh" | "gelu_new" | "gelu_tanh") => ActivationSpec::GeluTanh,
        // gpt-oss fixes its clamped gate's sigmoid slope; only the limit is
        // configured.
        Some("silu" | "swish") | None => {
            decoder.swiglu_limit.map_or(ActivationSpec::SwiGlu, |limit| {
                ActivationSpec::ClampedSwiGlu { alpha: 1.702, limit }
            })
        },
        Some(name) => ActivationSpec::NamedGated { name: name.to_owned() },
    }
}
