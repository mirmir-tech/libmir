use models::weights::{BlockQuantization, TensorBinding, TensorStorage};

use super::tensor;
use crate::{
    CudaBackend, CudaTensorSet, DenseDownSource, DenseFeedForwardSource, DenseGateUpSource,
    DenseOutputSource, DenseQkvSource, DenseSwiGluLayerTemplate, Error, NvFp4Config,
    NvFp4LinearWeight, NvFp4ScaleMode, NvFp4Tensors, ProjectionFormat, Result,
    checkpoint::dense::DenseLayerBindings,
};

/// NVFP4 feed-forward projections with NVFP4 or BF16 attention projections;
/// `ModelOpt` checkpoints such as Gemma 4 31B keep attention in BF16.
pub(super) fn load(
    backend: &CudaBackend,
    mut block: crate::DenseSwiGluConfig,
    tensors: &CudaTensorSet,
    bindings: DenseLayerBindings<'_>,
) -> Result<DenseSwiGluLayerTemplate> {
    let feed_forward = bindings.feed_forward.gated()?;
    let hidden = block.attention.hidden_size;
    let intermediate = block.intermediate_size;
    let gate = weight(backend, tensors, feed_forward.gate, hidden, intermediate)?;
    let up = weight(backend, tensors, feed_forward.up, hidden, intermediate)?;
    let down = weight(backend, tensors, feed_forward.down, intermediate, hidden)?;
    let feed_forward = DenseFeedForwardSource::Gated {
        gate_up: DenseGateUpSource::NvFp4 { gate: &gate, up: &up },
        down: DenseDownSource::NvFp4(&down),
    };
    let attention = bindings.attention;
    if matches!(attention.query.storage, TensorStorage::Dense { .. }) {
        block.attention.projection_format = ProjectionFormat::Bf16;
        let qkv = backend.pack_bf16_linears([
            tensor(tensors, &attention.query.source)?,
            tensor(tensors, &attention.key.source)?,
            tensor(tensors, &attention.value.source)?,
        ])?;
        let output = DenseOutputSource::Bf16(tensor(tensors, &attention.output.source)?);
        let source = super::common_source(
            tensors,
            bindings,
            DenseQkvSource::Bf16(&qkv),
            output,
            feed_forward,
        )?;
        return backend.prepare_dense_swiglu_layer_template(block, source);
    }
    let head = block.attention.cache.key_head_dim;
    let query = block.attention.query_heads * head;
    let key_value = block.attention.cache.kv_heads * head;
    let q = weight(backend, tensors, attention.query, hidden, query)?;
    let k = weight(backend, tensors, attention.key, hidden, key_value)?;
    let v = weight(backend, tensors, attention.value, hidden, key_value)?;
    let output = weight(backend, tensors, attention.output, query, hidden)?;
    let source = super::common_source(
        tensors,
        bindings,
        DenseQkvSource::NvFp4([&q, &k, &v]),
        DenseOutputSource::NvFp4(&output),
        feed_forward,
    )?;
    backend.prepare_dense_swiglu_layer_template(block, source)
}

fn weight(
    backend: &CudaBackend,
    tensors: &CudaTensorSet,
    binding: &TensorBinding,
    input_features: usize,
    output_features: usize,
) -> Result<NvFp4LinearWeight> {
    let TensorStorage::BlockQuantized {
        format: BlockQuantization::NVFP4,
        scales,
        global_scale: Some(global_scale),
        input_scale: Some(input_scale),
        ..
    } = &binding.storage
    else {
        return Err(Error::InvalidNvFp4("dense projection has no complete NVFP4 binding"));
    };
    backend.prepare_nvfp4_linear_weight(
        NvFp4Config::new(input_features, output_features),
        NvFp4Tensors {
            weight: tensor(tensors, &binding.source)?,
            weight_scale: tensor(tensors, scales)?,
            weight_scale_2: tensor(tensors, global_scale)?,
            input_scale: tensor(tensors, input_scale)?,
            scale_mode: NvFp4ScaleMode::from_names(global_scale, input_scale)?,
        },
    )
}
