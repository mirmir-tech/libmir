use super::super::super::super::{
    CudaBackend, CudaTensorSet, DenseDownSource, DenseFeedForwardSource, DenseGateUpSource,
    DenseOutputSource, DenseQkvSource, DenseSwiGluLayerTemplate, DirectFp8CheckpointWeight, Result,
};
use crate::checkpoint::dense::DenseLayerBindings;

pub(super) fn load(
    backend: &CudaBackend,
    block: crate::DenseSwiGluConfig,
    tensors: &CudaTensorSet,
    bindings: DenseLayerBindings<'_>,
) -> Result<DenseSwiGluLayerTemplate> {
    let feed_forward = bindings.feed_forward.gated()?;
    let fp8 = |binding| DirectFp8CheckpointWeight::load_binding(tensors, binding);
    let q = fp8(bindings.attention.query)?;
    let k = fp8(bindings.attention.key)?;
    let v = fp8(bindings.attention.value)?;
    let output = fp8(bindings.attention.output)?;
    let gate = fp8(feed_forward.gate)?;
    let up = fp8(feed_forward.up)?;
    let down = fp8(feed_forward.down)?;
    backend.prepare_dense_swiglu_layer_template(
        block,
        super::common_source(
            tensors,
            bindings,
            DenseQkvSource::DirectFp8([&q, &k, &v]),
            DenseOutputSource::DirectFp8(&output),
            DenseFeedForwardSource::Gated {
                gate_up: DenseGateUpSource::DirectFp8 { gate: &gate, up: &up },
                down: DenseDownSource::DirectFp8(&down),
            },
        )?,
    )
}
