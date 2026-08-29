use models::{
    layout::DecoderConfig,
    weights::{
        BlockQuantization, DenseDecoderLayerBindings, TensorBinding, TensorCatalog, TensorStorage,
    },
};

use super::source::DenseLayerSource;
use crate::{
    CudaBackend, CudaTensor, CudaTensorSet, DenseDownSource, DenseGateUpSource, DenseOutputSource,
    DenseQkvSource, DenseSwiGluLayerLoadConfig, DenseSwiGluLayerTemplate, DenseWeightSource, Error,
    NvFp4Config, NvFp4LinearWeight, NvFp4ScaleMode, NvFp4Tensors, PackedIntegerWeight,
    ProjectionFormat, Result, checkpoint::model::payload_bytes,
};

mod direct_fp8;
mod mxfp4;
mod mxfp8;

impl CudaBackend {
    pub(crate) fn load_dense_swiglu_layer_tracked(
        &self,
        decoder: &DecoderConfig,
        catalog: &TensorCatalog,
        layer: usize,
        bindings: DenseDecoderLayerBindings<'_>,
        load: DenseSwiGluLayerLoadConfig,
    ) -> Result<(DenseSwiGluLayerTemplate, u64)> {
        let block = load.block(decoder, layer)?;
        let source = DenseLayerSource::discover(catalog, bindings)?;
        let tensors = source.upload(self, bindings, load.projection_format)?;
        let template = match load.projection_format {
            ProjectionFormat::Affine => load_affine(self, block, &tensors, bindings)?,
            ProjectionFormat::Bf16 => load_bf16(self, block, &tensors, bindings)?,
            ProjectionFormat::DirectFp8 => direct_fp8::load(self, block, &tensors, bindings)?,
            ProjectionFormat::MxFp4 => mxfp4::load(self, block, &tensors, bindings)?,
            ProjectionFormat::MxFp8 => mxfp8::load(self, block, &tensors, bindings)?,
            ProjectionFormat::PackedInteger => {
                load_packed_integer(self, block, &tensors, bindings)?
            },
            ProjectionFormat::NvFp4 => load_nvfp4(self, block, &tensors, bindings)?,
        };
        tracing::debug!(
            backend = "cuda",
            layer,
            tensors = source.tensors.len(),
            format = ?load.projection_format,
            "loaded dense SwiGLU layer template"
        );
        Ok((template, payload_bytes(source.tensors)?))
    }
}

fn load_affine(
    backend: &CudaBackend,
    block: crate::DenseSwiGluConfig,
    tensors: &CudaTensorSet,
    bindings: DenseDecoderLayerBindings<'_>,
) -> Result<DenseSwiGluLayerTemplate> {
    let affine = |binding| crate::AffineQuantizedWeight::load_binding(tensors, binding);
    let q = affine(bindings.attention.query)?;
    let k = affine(bindings.attention.key)?;
    let v = affine(bindings.attention.value)?;
    let output = affine(bindings.attention.output)?;
    let gate = affine(bindings.gate)?;
    let up = affine(bindings.up)?;
    let down = affine(bindings.down)?;
    backend.prepare_dense_swiglu_layer_template(
        block,
        common_source(
            tensors,
            bindings,
            DenseQkvSource::Affine([&q, &k, &v]),
            DenseOutputSource::Affine(&output),
            DenseGateUpSource::Affine { gate: &gate, up: &up },
            DenseDownSource::Affine(&down),
        )?,
    )
}

fn load_packed_integer(
    backend: &CudaBackend,
    block: crate::DenseSwiGluConfig,
    tensors: &CudaTensorSet,
    bindings: DenseDecoderLayerBindings<'_>,
) -> Result<DenseSwiGluLayerTemplate> {
    let hidden = block.attention.hidden_size;
    let head = block.attention.cache.key_head_dim;
    let query = block.attention.query_heads * head;
    let key_value = block.attention.cache.kv_heads * head;
    let intermediate = block.intermediate_size;
    let q = packed_integer(tensors, bindings.attention.query, hidden, query)?;
    let k = packed_integer(tensors, bindings.attention.key, hidden, key_value)?;
    let v = packed_integer(tensors, bindings.attention.value, hidden, key_value)?;
    let output = packed_integer(tensors, bindings.attention.output, query, hidden)?;
    let gate = packed_integer(tensors, bindings.gate, hidden, intermediate)?;
    let up = packed_integer(tensors, bindings.up, hidden, intermediate)?;
    let down = packed_integer(tensors, bindings.down, intermediate, hidden)?;
    backend.prepare_dense_swiglu_layer_template(
        block,
        common_source(
            tensors,
            bindings,
            DenseQkvSource::PackedInteger([&q, &k, &v]),
            DenseOutputSource::PackedInteger(&output),
            DenseGateUpSource::PackedInteger { gate: &gate, up: &up },
            DenseDownSource::PackedInteger(&down),
        )?,
    )
}

fn packed_integer(
    tensors: &CudaTensorSet,
    binding: &TensorBinding,
    input: usize,
    output: usize,
) -> Result<PackedIntegerWeight> {
    PackedIntegerWeight::load_binding(tensors, binding, input, output)
}

fn load_bf16(
    backend: &CudaBackend,
    block: crate::DenseSwiGluConfig,
    tensors: &CudaTensorSet,
    bindings: DenseDecoderLayerBindings<'_>,
) -> Result<DenseSwiGluLayerTemplate> {
    let qkv = backend.pack_bf16_linears([
        tensor(tensors, &bindings.attention.query.source)?,
        tensor(tensors, &bindings.attention.key.source)?,
        tensor(tensors, &bindings.attention.value.source)?,
    ])?;
    let gate_up = backend.pack_bf16_linear_pair(
        tensor(tensors, &bindings.gate.source)?,
        tensor(tensors, &bindings.up.source)?,
    )?;
    backend.prepare_dense_swiglu_layer_template(
        block,
        common_source(
            tensors,
            bindings,
            DenseQkvSource::Bf16(&qkv),
            DenseOutputSource::Bf16(tensor(tensors, &bindings.attention.output.source)?),
            DenseGateUpSource::Bf16(&gate_up),
            DenseDownSource::Bf16(tensor(tensors, &bindings.down.source)?),
        )?,
    )
}

fn load_nvfp4(
    backend: &CudaBackend,
    block: crate::DenseSwiGluConfig,
    tensors: &CudaTensorSet,
    bindings: DenseDecoderLayerBindings<'_>,
) -> Result<DenseSwiGluLayerTemplate> {
    let hidden = block.attention.hidden_size;
    let head = block.attention.cache.key_head_dim;
    let query = block.attention.query_heads * head;
    let key_value = block.attention.cache.kv_heads * head;
    let intermediate = block.intermediate_size;
    let q = nvfp4_weight(backend, tensors, bindings.attention.query, hidden, query)?;
    let k = nvfp4_weight(backend, tensors, bindings.attention.key, hidden, key_value)?;
    let v = nvfp4_weight(backend, tensors, bindings.attention.value, hidden, key_value)?;
    let output = nvfp4_weight(backend, tensors, bindings.attention.output, query, hidden)?;
    let gate = nvfp4_weight(backend, tensors, bindings.gate, hidden, intermediate)?;
    let up = nvfp4_weight(backend, tensors, bindings.up, hidden, intermediate)?;
    let down = nvfp4_weight(backend, tensors, bindings.down, intermediate, hidden)?;
    backend.prepare_dense_swiglu_layer_template(
        block,
        common_source(
            tensors,
            bindings,
            DenseQkvSource::NvFp4([&q, &k, &v]),
            DenseOutputSource::NvFp4(&output),
            DenseGateUpSource::NvFp4 { gate: &gate, up: &up },
            DenseDownSource::NvFp4(&down),
        )?,
    )
}

pub(super) fn common_source<'a>(
    tensors: &'a CudaTensorSet,
    bindings: DenseDecoderLayerBindings<'_>,
    qkv: DenseQkvSource<'a>,
    output: DenseOutputSource<'a>,
    gate_up: DenseGateUpSource<'a>,
    down: DenseDownSource<'a>,
) -> Result<DenseWeightSource<'a>> {
    Ok(DenseWeightSource {
        input_norm: tensor(tensors, &bindings.input_norm.source)?,
        qkv,
        query_norm: optional_tensor(tensors, bindings.attention.query_norm)?,
        key_norm: optional_tensor(tensors, bindings.attention.key_norm)?,
        output,
        post_attention_norm: tensor(tensors, &bindings.post_attention_norm.source)?,
        gate_up,
        down,
    })
}

fn nvfp4_weight(
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

fn optional_tensor<'a>(
    tensors: &'a CudaTensorSet,
    binding: Option<&TensorBinding>,
) -> Result<Option<&'a CudaTensor>> {
    binding.map(|binding| tensor(tensors, &binding.source)).transpose()
}

fn tensor<'a>(tensors: &'a CudaTensorSet, name: &str) -> Result<&'a CudaTensor> {
    tensors.get(name).ok_or_else(|| Error::MissingTensor(name.into()))
}
