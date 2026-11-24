//! Projection output biases (Qwen2, GLM-4 attention). GGML projections add
//! them after their matmul; the other dense formats have no bias epilogue
//! yet, so a layer that declares one is refused rather than run without it.
//! Norm biases are packed with their weights instead (`norm_bias`).

use models::weights::TensorStorage;

use crate::{Error, ProjectionFormat, Result, checkpoint::dense::DenseLayerBindings};

pub(super) fn require_applied(
    bindings: DenseLayerBindings<'_>,
    format: ProjectionFormat,
) -> Result<()> {
    if format == ProjectionFormat::Ggml {
        return Ok(());
    }
    let norms = bindings.norms();
    let projections = bindings.bindings().into_iter().filter(|binding| !norms.contains(binding));
    let biased = projections.into_iter().any(|binding| {
        matches!(
            &binding.storage,
            TensorStorage::Dense { bias: Some(_), .. }
                | TensorStorage::Ggml {
                    companion: models::weights::GgmlCompanion::Bias(_),
                    ..
                }
        )
    });
    if biased {
        return Err(Error::UnsupportedDecoderLayer(format!(
            "CUDA adds projection biases only to GGML dense layers, not {format:?}"
        )));
    }
    Ok(())
}
