//! Activations the semantic spec could not resolve, which no backend runs.

use models::semantic::{ActivationSpec, FeedForwardSpec, SemanticModelSpec};

use crate::{ArchitectureError, Result};

/// Refuses a decoder with an activation the semantic spec kept only by name,
/// before a backend loads weights it cannot run.
pub fn require_resolved(semantic: &SemanticModelSpec) -> Result<()> {
    let decoder = &semantic.decoder;
    for layer in decoder.layers.iter().chain(&decoder.prediction_layers) {
        if let Some(name) = activations(&layer.feed_forward).find_map(|activation| match activation
        {
            ActivationSpec::NamedGated { name } => Some(name),
            _ => None,
        }) {
            return Err(ArchitectureError::invalid(format!(
                "no backend runs the {name} activation of layer {}",
                layer.index
            )));
        }
    }
    Ok(())
}

fn activations(feed_forward: &FeedForwardSpec) -> impl Iterator<Item = &ActivationSpec> {
    let (first, second) = match feed_forward {
        FeedForwardSpec::Dense { activation, .. } => (Some(activation), None),
        FeedForwardSpec::Routed { routed, shared }
        | FeedForwardSpec::LatentRouted { routed, shared, .. } => {
            (Some(&routed.activation), shared.as_ref().map(|shared| &shared.activation))
        },
        FeedForwardSpec::DenseAndRouted { dense_activation, routed, .. } => {
            (Some(dense_activation), Some(&routed.activation))
        },
        FeedForwardSpec::None => (None, None),
    };
    first.into_iter().chain(second)
}
