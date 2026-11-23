//! The gated activation a decoder configuration names.

use crate::{layout::DecoderConfig, semantic::ActivationSpec};

/// The gated activation of `hidden_act`, its aliases resolved once here:
/// transformers' `gelu_new` is the tanh approximation, `swish` is `SiLU`, and
/// a missing name defaults to `SiLU` as Llama's configuration does.
pub(super) fn activation(decoder: &DecoderConfig) -> ActivationSpec {
    match decoder.hidden_activation.as_deref() {
        Some("gelu_pytorch_tanh" | "gelu_new" | "gelu_tanh") => ActivationSpec::GeluTanh,
        Some("silu" | "swish") | None => ActivationSpec::SwiGlu {
            alpha: if decoder.swiglu_limit.is_some() {
                1.702
            } else {
                1.0
            },
            clamp: decoder.swiglu_limit,
            up_shift: if decoder.swiglu_limit.is_some() {
                1.0
            } else {
                0.0
            },
        },
        Some(name) => ActivationSpec::NamedGated { name: name.to_owned() },
    }
}
