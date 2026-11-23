use super::super::super::super::{
    CudaBackend, CudaTensorSet, DenseDownSource, DenseFeedForwardSource, DenseGateUpSource,
    DenseOutputSource, DenseQkvSource, DenseSwiGluLayerTemplate, GgmlCheckpointWeight, Result,
};
use crate::checkpoint::dense::DenseLayerBindings;

pub(super) fn load(
    backend: &CudaBackend,
    block: crate::DenseSwiGluConfig,
    tensors: &CudaTensorSet,
    bindings: DenseLayerBindings<'_>,
    residual: Option<f32>,
) -> Result<DenseSwiGluLayerTemplate> {
    let feed_forward = bindings.feed_forward.gated()?;
    // Dense layers add their projections' output biases (Qwen2, GLM-4).
    let weight = |binding| GgmlCheckpointWeight::load_binding_with_bias(tensors, binding);
    let q = weight(bindings.attention.query)?;
    let k = weight(bindings.attention.key)?;
    let v = weight(bindings.attention.value)?;
    // The residual multiplier ends both branches (`residual`).
    let branch = |binding| match residual {
        Some(factor) => weight(binding)?.scaled(backend, factor),
        None => weight(binding),
    };
    let output = branch(bindings.attention.output)?;
    let gate = weight(feed_forward.gate)?;
    let up = weight(feed_forward.up)?;
    let down = branch(feed_forward.down)?;
    backend.prepare_dense_swiglu_layer_template(
        block,
        super::common_source(
            tensors,
            bindings,
            DenseQkvSource::Ggml([&q, &k, &v]),
            DenseOutputSource::Ggml(&output),
            DenseFeedForwardSource::Gated {
                gate_up: DenseGateUpSource::Ggml { gate: &gate, up: &up },
                down: DenseDownSource::Ggml(&down),
            },
        )?,
    )
}
