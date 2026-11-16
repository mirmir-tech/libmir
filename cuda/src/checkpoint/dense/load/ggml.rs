use models::weights::DenseDecoderLayerBindings;

use super::super::super::super::{
    CudaBackend, CudaTensorSet, DenseDownSource, DenseFeedForwardSource, DenseGateUpSource,
    DenseOutputSource, DenseQkvSource, DenseSwiGluLayerTemplate, GgmlCheckpointWeight, Result,
};

pub(super) fn load(
    backend: &CudaBackend,
    block: crate::DenseSwiGluConfig,
    tensors: &CudaTensorSet,
    bindings: DenseDecoderLayerBindings<'_>,
) -> Result<DenseSwiGluLayerTemplate> {
    let feed_forward = bindings.feed_forward.gated()?;
    let weight = |binding| GgmlCheckpointWeight::load_binding(tensors, binding);
    let q = weight(bindings.attention.query)?;
    let k = weight(bindings.attention.key)?;
    let v = weight(bindings.attention.value)?;
    let output = weight(bindings.attention.output)?;
    let gate = weight(feed_forward.gate)?;
    let up = weight(feed_forward.up)?;
    let down = weight(feed_forward.down)?;
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
