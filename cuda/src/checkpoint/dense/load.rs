use models::{layout::DecoderConfig, weights::TensorCatalog};

use super::source::{DenseLayerSource, nvfp4_auxiliary_names};
use crate::{
    CudaBackend, CudaTensor, CudaTensorSet, DenseDownSource, DenseGateUpSource, DenseOutputSource,
    DenseQkvSource, DenseSwiGluLayerLoadConfig, DenseSwiGluLayerTemplate, DenseWeightSource, Error,
    NvFp4Config, NvFp4LinearWeight, NvFp4Tensors, ProjectionFormat, Result,
    checkpoint::model::payload_bytes,
};

impl CudaBackend {
    pub(crate) fn load_dense_swiglu_layer_tracked(
        &self,
        decoder: &DecoderConfig,
        catalog: &TensorCatalog,
        layer: usize,
        load: DenseSwiGluLayerLoadConfig,
    ) -> Result<(DenseSwiGluLayerTemplate, u64)> {
        let block = load.block(decoder, layer)?;
        let source = DenseLayerSource::discover(
            catalog,
            layer,
            load.qkv_normalization,
            load.projection_format,
        )?;
        let tensors = upload(self, &source)?;
        let names = &source.names;
        let template = match load.projection_format {
            ProjectionFormat::Bf16 => load_bf16(self, block, &tensors, names)?,
            ProjectionFormat::NvFp4 => load_nvfp4(self, block, &tensors, names)?,
        };
        tracing::debug!(
            backend = "cuda",
            layer,
            prefix = source.prefix,
            tensors = source.tensors.len(),
            format = ?load.projection_format,
            "loaded dense SwiGLU layer template"
        );
        Ok((template, payload_bytes(source.tensors)?))
    }
}

fn load_bf16(
    backend: &CudaBackend,
    block: crate::DenseSwiGluConfig,
    tensors: &CudaTensorSet,
    names: &super::source::DenseLayerNames,
) -> Result<DenseSwiGluLayerTemplate> {
    let get = |name: &str| tensor(tensors, name);
    let qkv = backend.pack_bf16_linears([
        get(&names.required[1])?,
        get(&names.required[2])?,
        get(&names.required[3])?,
    ])?;
    let gate_up =
        backend.pack_bf16_linear_pair(get(&names.required[6])?, get(&names.required[7])?)?;
    backend.prepare_dense_swiglu_layer_template(
        block,
        common_source(
            tensors,
            names,
            DenseQkvSource::Bf16(&qkv),
            DenseOutputSource::Bf16(get(&names.required[4])?),
            DenseGateUpSource::Bf16(&gate_up),
            DenseDownSource::Bf16(get(&names.down)?),
        )?,
    )
}

fn load_nvfp4(
    backend: &CudaBackend,
    block: crate::DenseSwiGluConfig,
    tensors: &CudaTensorSet,
    names: &super::source::DenseLayerNames,
) -> Result<DenseSwiGluLayerTemplate> {
    let hidden = block.attention.hidden_size;
    let head = block.attention.cache.key_head_dim;
    let query = block.attention.query_heads * head;
    let key_value = block.attention.cache.kv_heads * head;
    let intermediate = block.intermediate_size;
    let q = nvfp4_weight(backend, tensors, &names.required[1], hidden, query)?;
    let k = nvfp4_weight(backend, tensors, &names.required[2], hidden, key_value)?;
    let v = nvfp4_weight(backend, tensors, &names.required[3], hidden, key_value)?;
    let output = nvfp4_weight(backend, tensors, &names.required[4], query, hidden)?;
    let gate = nvfp4_weight(backend, tensors, &names.required[6], hidden, intermediate)?;
    let up = nvfp4_weight(backend, tensors, &names.required[7], hidden, intermediate)?;
    let down = nvfp4_weight(backend, tensors, &names.down, intermediate, hidden)?;
    backend.prepare_dense_swiglu_layer_template(
        block,
        common_source(
            tensors,
            names,
            DenseQkvSource::NvFp4([&q, &k, &v]),
            DenseOutputSource::NvFp4(&output),
            DenseGateUpSource::NvFp4 { gate: &gate, up: &up },
            DenseDownSource::NvFp4(&down),
        )?,
    )
}

fn common_source<'a>(
    tensors: &'a CudaTensorSet,
    names: &super::source::DenseLayerNames,
    qkv: DenseQkvSource<'a>,
    output: DenseOutputSource<'a>,
    gate_up: DenseGateUpSource<'a>,
    down: DenseDownSource<'a>,
) -> Result<DenseWeightSource<'a>> {
    Ok(DenseWeightSource {
        input_norm: tensor(tensors, &names.required[0])?,
        qkv,
        query_norm: optional_tensor(tensors, names.query_norm.as_deref())?,
        key_norm: optional_tensor(tensors, names.key_norm.as_deref())?,
        output,
        post_attention_norm: tensor(tensors, &names.required[5])?,
        gate_up,
        down,
    })
}

fn nvfp4_weight(
    backend: &CudaBackend,
    tensors: &CudaTensorSet,
    name: &str,
    input_features: usize,
    output_features: usize,
) -> Result<NvFp4LinearWeight> {
    let [weight_scale, weight_scale_2, input_scale] = nvfp4_auxiliary_names(name)?;
    backend.prepare_nvfp4_linear_weight(
        NvFp4Config::new(input_features, output_features),
        NvFp4Tensors {
            weight: tensor(tensors, name)?,
            weight_scale: tensor(tensors, &weight_scale)?,
            weight_scale_2: tensor(tensors, &weight_scale_2)?,
            input_scale: tensor(tensors, &input_scale)?,
        },
    )
}

fn upload(backend: &CudaBackend, source: &DenseLayerSource<'_>) -> Result<CudaTensorSet> {
    let mut upload = backend.begin_tensor_upload();
    for tensor in &source.tensors {
        upload.enqueue(tensor)?;
    }
    upload.finish()
}

fn optional_tensor<'a>(
    tensors: &'a CudaTensorSet,
    name: Option<&str>,
) -> Result<Option<&'a CudaTensor>> {
    name.map(|name| tensor(tensors, name)).transpose()
}

fn tensor<'a>(tensors: &'a CudaTensorSet, name: &str) -> Result<&'a CudaTensor> {
    tensors.get(name).ok_or_else(|| Error::MissingTensor(name.into()))
}
