//! Granite's residual multiplier: each residual branch is scaled before it
//! joins the residual stream. The branches end in the attention output and
//! the feed-forward down projections (a routed layer's experts and its shared
//! expert), so the factor folds into them at load, `x + m (W a) = x + (m W)
//! a`: dense weights are scaled here, in place, and GGML ones carry it as
//! their F32 output scale, which their loaders set.

use models::{
    semantic::SemanticModelSpec,
    weights::{
        DenseLayerFeedForwardBindings, ExpertProjectionRole, RoutedExpertBindings, TensorBinding,
        TensorStorage,
    },
};

use crate::{CudaTensorSet, Error, Result, checkpoint::dense::DenseLayerBindings};

/// The factor of every residual branch when it is not one.
pub(super) fn factor(semantic: &SemanticModelSpec, bias: bool) -> Result<Option<f32>> {
    let multiplier = semantic.decoder.residual_multiplier;
    #[allow(clippy::float_cmp)]
    if multiplier == 1.0 {
        return Ok(None);
    }
    // A bias after the projection would escape the folded factor.
    if bias {
        return Err(unsupported("a residual multiplier with attention biases"));
    }
    Ok(Some(multiplier.to_string().parse()?))
}

/// Scales every dense projection that ends a residual branch by `factor`;
/// the GGML loaders fold it into their weights' output scales. Each binding's
/// storage decides, so a layer mixing dense and GGML tensors folds each once.
pub(super) fn fold_dense(
    tensors: &CudaTensorSet,
    bindings: DenseLayerBindings<'_>,
    factor: f32,
) -> Result<()> {
    for binding in branch_ends(bindings) {
        match &binding.storage {
            TensorStorage::Dense { bias: None, .. } => {
                tensors.scale_bf16(&binding.source, factor)?;
            },
            TensorStorage::Ggml { .. } => {},
            TensorStorage::Dense { bias: Some(_), .. } => {
                return Err(unsupported("a residual multiplier on a biased projection"));
            },
            _ => {
                return Err(Error::UnsupportedDecoderLayer(format!(
                    "CUDA folds a residual multiplier only into dense and GGML projections: {}",
                    binding.source
                )));
            },
        }
    }
    Ok(())
}

/// The projections whose products join the residual stream.
fn branch_ends(bindings: DenseLayerBindings<'_>) -> Vec<&TensorBinding> {
    let mut ends = vec![bindings.attention.output()];
    match bindings.feed_forward {
        DenseLayerFeedForwardBindings::Gated(feed_forward) => ends.push(feed_forward.down),
        DenseLayerFeedForwardBindings::Routed(routed) => ends.extend(expert_downs(routed.experts)),
        DenseLayerFeedForwardBindings::SharedRouted(shared) => {
            ends.extend(expert_downs(shared.routed));
            ends.push(shared.shared_down);
        },
    }
    ends
}

fn expert_downs(experts: RoutedExpertBindings<'_>) -> Vec<&TensorBinding> {
    match experts {
        RoutedExpertBindings::InterleavedGateUp { down, .. }
        | RoutedExpertBindings::SeparateGateUp { down, .. }
        | RoutedExpertBindings::UngatedStacked { down, .. } => vec![down],
        RoutedExpertBindings::Individual { .. } => experts.individual(ExpertProjectionRole::Down),
    }
}

fn unsupported(what: &str) -> Error {
    Error::UnsupportedDecoderLayer(format!("CUDA has no {what}"))
}

#[cfg(test)]
mod tests;
