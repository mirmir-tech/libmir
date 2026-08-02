use models::weights::DenseDecoderLayerBindings;

use super::super::super::super::{
    CudaBackend, CudaTensorSet, DenseDownSource, DenseGateUpSource, DenseOutputSource,
    DenseQkvSource, DenseSwiGluLayerTemplate, MxFp4CheckpointWeight, Result,
};

pub(super) fn load(
    backend: &CudaBackend,
    block: crate::DenseSwiGluConfig,
    tensors: &CudaTensorSet,
    bindings: DenseDecoderLayerBindings<'_>,
) -> Result<DenseSwiGluLayerTemplate> {
    let weight = |binding| MxFp4CheckpointWeight::load_binding(tensors, binding);
    let q = weight(bindings.attention.query)?;
    let k = weight(bindings.attention.key)?;
    let v = weight(bindings.attention.value)?;
    let output = weight(bindings.attention.output)?;
    let gate = weight(bindings.gate)?;
    let up = weight(bindings.up)?;
    let down = weight(bindings.down)?;
    backend.prepare_dense_swiglu_layer_template(
        block,
        super::common_source(
            tensors,
            bindings,
            DenseQkvSource::MxFp4([&q, &k, &v]),
            DenseOutputSource::MxFp4(&output),
            DenseGateUpSource::MxFp4 { gate: &gate, up: &up },
            DenseDownSource::MxFp4(&down),
        )?,
    )
}
