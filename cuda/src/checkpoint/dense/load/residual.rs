//! Granite's residual multiplier: each residual branch is scaled before it
//! joins the residual stream. The branches end in the attention output and
//! the feed-forward down projections, so the factor folds into them at load,
//! `x + m (W a) = x + (m W) a`: BF16 weights are scaled, GGML ones carry it
//! as their F32 output scale.

use models::{semantic::SemanticModelSpec, weights::DenseLayerFeedForwardBindings};

use crate::{
    CudaTensorSet, Error, ProjectionFormat, Result, checkpoint::dense::DenseLayerBindings,
};

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

/// Folds `factor` into the BF16 branch projections; GGML loaders fold it
/// into their weights' output scales.
pub(super) fn fold_dense(
    tensors: &CudaTensorSet,
    bindings: DenseLayerBindings<'_>,
    format: ProjectionFormat,
    factor: f32,
) -> Result<()> {
    match (format, bindings.feed_forward) {
        (ProjectionFormat::Ggml, _) => Ok(()),
        (ProjectionFormat::Bf16, DenseLayerFeedForwardBindings::Gated(feed_forward)) => {
            tensors.scale_bf16(&bindings.attention.output().source, factor)?;
            tensors.scale_bf16(&feed_forward.down.source, factor)
        },
        _ => Err(unsupported("a residual multiplier outside BF16 and GGML dense layers")),
    }
}

fn unsupported(what: &str) -> Error {
    Error::UnsupportedDecoderLayer(format!("CUDA has no {what}"))
}
