//! Hardware-free model architecture admission shared by libmir backends.

mod activation;
mod error;
pub mod ggml;

pub mod cuda;
pub mod metal;

pub use error::{ArchitectureError, Result};

#[cfg(test)]
#[allow(clippy::self_named_module_files)]
mod tests;

/// Whether the decoder has Gemma 4's E-model features: per-layer inputs, or
/// layers attending over an earlier layer's keys and values.
#[must_use]
pub(crate) fn uses_layer_sharing(spec: &models::semantic::SemanticModelSpec) -> bool {
    spec.decoder.per_layer_input.is_some()
        || spec.decoder.layers.iter().any(|layer| {
            layer
                .mixer
                .softmax_attention()
                .is_some_and(|attention| attention.key_value_source.is_some())
        })
}
