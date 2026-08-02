use models::weights::DenseDecoderLayerBindings;

use super::super::super::super::{
    CudaBackend, CudaTensorSet, DenseDownSource, DenseGateUpSource, DenseOutputSource,
    DenseQkvSource, DenseSwiGluLayerTemplate, DirectFp8CheckpointWeight, Result,
};

pub(super) fn load(
    backend: &CudaBackend,
    block: crate::DenseSwiGluConfig,
    tensors: &CudaTensorSet,
    bindings: DenseDecoderLayerBindings<'_>,
) -> Result<DenseSwiGluLayerTemplate> {
    let fp8 = |binding| DirectFp8CheckpointWeight::load_binding(tensors, binding);
    let q = fp8(bindings.attention.query)?;
    let k = fp8(bindings.attention.key)?;
    let v = fp8(bindings.attention.value)?;
    let output = fp8(bindings.attention.output)?;
    let gate = fp8(bindings.gate)?;
    let up = fp8(bindings.up)?;
    let down = fp8(bindings.down)?;
    backend.prepare_dense_swiglu_layer_template(
        block,
        super::common_source(
            tensors,
            bindings,
            DenseQkvSource::DirectFp8([&q, &k, &v]),
            DenseOutputSource::DirectFp8(&output),
            DenseGateUpSource::DirectFp8 { gate: &gate, up: &up },
            DenseDownSource::DirectFp8(&down),
        )?,
    )
}
