use models::{
    layout::DecoderConfig,
    semantic::SemanticModelSpec,
    weights::{
        DenseDecoderLayerBindings, DenseLayerFeedForwardBindings, TensorBinding, TensorCatalog,
    },
};

use super::source::DenseLayerSource;
use crate::{
    CudaBackend, CudaTensor, CudaTensorSet, DenseDownSource, DenseFeedForwardSource,
    DenseGateUpSource, DenseOutputSource, DenseQkvSource, DenseSandwichWeights,
    DenseSwiGluLayerLoadConfig, DenseSwiGluLayerTemplate, DenseWeightSource, Error,
    PackedIntegerWeight, ProjectionFormat, Result, checkpoint::model::payload_bytes,
};

mod direct_fp8;
mod ggml;
mod mxfp4;
mod mxfp8;
mod nvfp4;
mod rope_rows;
mod routed;

impl CudaBackend {
    pub(crate) fn load_dense_swiglu_layer_tracked(
        &self,
        decoder: &DecoderConfig,
        semantic: &SemanticModelSpec,
        catalog: &TensorCatalog,
        layer: usize,
        bindings: DenseDecoderLayerBindings<'_>,
        load: DenseSwiGluLayerLoadConfig,
    ) -> Result<(DenseSwiGluLayerTemplate, u64)> {
        let block = load.block(decoder, semantic, layer)?;
        let source = DenseLayerSource::discover(catalog, bindings)?;
        let mut tensors = source.upload(self, bindings, load.projection_format)?;
        if decoder.query_key_rows_interleaved {
            rope_rows::restore(self, &mut tensors, bindings, block.attention.cache.key_head_dim)?;
        }
        let offset = semantic
            .decoder
            .layers
            .get(layer)
            .map_or(0.0, |spec| spec.input_norm.weight_offset);
        if offset != 0.0 {
            let attention = bindings.attention;
            let mut norms = vec![bindings.input_norm, bindings.post_attention_norm];
            norms.extend(attention.query_norm.into_iter().chain(attention.key_norm));
            if let Some(sandwich) = bindings.sandwich {
                norms.extend([sandwich.attention_output_norm, sandwich.feed_forward_output_norm]);
            }
            for norm in norms {
                tensors.add_bf16(&norm.source, offset)?;
            }
        }
        let template = if let DenseLayerFeedForwardBindings::Routed(routed) = bindings.feed_forward
        {
            routed::load(self, block, &tensors, bindings, routed)?
        } else {
            self.load_gated(block, &tensors, bindings, load.projection_format)?
        };
        tracing::debug!(
            backend = "cuda",
            layer,
            tensors = source.tensors.len(),
            format = ?load.projection_format,
            routed = block.experts.is_some(),
            "loaded dense SwiGLU layer template"
        );
        Ok((template, payload_bytes(source.tensors)?))
    }

    fn load_gated(
        &self,
        block: crate::DenseSwiGluConfig,
        tensors: &CudaTensorSet,
        bindings: DenseDecoderLayerBindings<'_>,
        format: ProjectionFormat,
    ) -> Result<DenseSwiGluLayerTemplate> {
        Ok(match format {
            ProjectionFormat::Affine => load_affine(self, block, tensors, bindings)?,
            ProjectionFormat::Bf16 => load_bf16(self, block, tensors, bindings)?,
            ProjectionFormat::DirectFp8 => direct_fp8::load(self, block, tensors, bindings)?,
            ProjectionFormat::Ggml => ggml::load(self, block, tensors, bindings)?,
            ProjectionFormat::MxFp4 => mxfp4::load(self, block, tensors, bindings)?,
            ProjectionFormat::MxFp8 => mxfp8::load(self, block, tensors, bindings)?,
            ProjectionFormat::PackedInteger => load_packed_integer(self, block, tensors, bindings)?,
            ProjectionFormat::NvFp4 => nvfp4::load(self, block, tensors, bindings)?,
        })
    }
}

fn load_affine(
    backend: &CudaBackend,
    block: crate::DenseSwiGluConfig,
    tensors: &CudaTensorSet,
    bindings: DenseDecoderLayerBindings<'_>,
) -> Result<DenseSwiGluLayerTemplate> {
    let feed_forward = bindings.feed_forward.gated()?;
    let affine = |binding| crate::AffineQuantizedWeight::load_binding(tensors, binding);
    let q = affine(bindings.attention.query)?;
    let k = affine(bindings.attention.key)?;
    let v = affine(bindings.attention.value)?;
    let output = affine(bindings.attention.output)?;
    let gate = affine(feed_forward.gate)?;
    let up = affine(feed_forward.up)?;
    let down = affine(feed_forward.down)?;
    backend.prepare_dense_swiglu_layer_template(
        block,
        common_source(
            tensors,
            bindings,
            DenseQkvSource::Affine([&q, &k, &v]),
            DenseOutputSource::Affine(&output),
            DenseFeedForwardSource::Gated {
                gate_up: DenseGateUpSource::Affine { gate: &gate, up: &up },
                down: DenseDownSource::Affine(&down),
            },
        )?,
    )
}

fn load_packed_integer(
    backend: &CudaBackend,
    block: crate::DenseSwiGluConfig,
    tensors: &CudaTensorSet,
    bindings: DenseDecoderLayerBindings<'_>,
) -> Result<DenseSwiGluLayerTemplate> {
    let feed_forward = bindings.feed_forward.gated()?;
    let hidden = block.attention.hidden_size;
    let head = block.attention.cache.key_head_dim;
    let query = block.attention.query_heads * head;
    let key_value = block.attention.cache.kv_heads * head;
    let intermediate = block.intermediate_size;
    let q = packed_integer(tensors, bindings.attention.query, hidden, query)?;
    let k = packed_integer(tensors, bindings.attention.key, hidden, key_value)?;
    let v = packed_integer(tensors, bindings.attention.value, hidden, key_value)?;
    let output = packed_integer(tensors, bindings.attention.output, query, hidden)?;
    let gate = packed_integer(tensors, feed_forward.gate, hidden, intermediate)?;
    let up = packed_integer(tensors, feed_forward.up, hidden, intermediate)?;
    let down = packed_integer(tensors, feed_forward.down, intermediate, hidden)?;
    backend.prepare_dense_swiglu_layer_template(
        block,
        common_source(
            tensors,
            bindings,
            DenseQkvSource::PackedInteger([&q, &k, &v]),
            DenseOutputSource::PackedInteger(&output),
            DenseFeedForwardSource::Gated {
                gate_up: DenseGateUpSource::PackedInteger { gate: &gate, up: &up },
                down: DenseDownSource::PackedInteger(&down),
            },
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
    let feed_forward = bindings.feed_forward.gated()?;
    let qkv = backend.pack_bf16_linears([
        tensor(tensors, &bindings.attention.query.source)?,
        tensor(tensors, &bindings.attention.key.source)?,
        tensor(tensors, &bindings.attention.value.source)?,
    ])?;
    let gate_up = backend.pack_bf16_linear_pair(
        tensor(tensors, &feed_forward.gate.source)?,
        tensor(tensors, &feed_forward.up.source)?,
    )?;
    backend.prepare_dense_swiglu_layer_template(
        block,
        common_source(
            tensors,
            bindings,
            DenseQkvSource::Bf16(&qkv),
            DenseOutputSource::Bf16(tensor(tensors, &bindings.attention.output.source)?),
            DenseFeedForwardSource::Gated {
                gate_up: DenseGateUpSource::Bf16(&gate_up),
                down: DenseDownSource::Bf16(tensor(tensors, &feed_forward.down.source)?),
            },
        )?,
    )
}

pub(super) fn common_source<'a>(
    tensors: &'a CudaTensorSet,
    bindings: DenseDecoderLayerBindings<'_>,
    qkv: DenseQkvSource<'a>,
    output: DenseOutputSource<'a>,
    feed_forward: DenseFeedForwardSource<'a>,
) -> Result<DenseWeightSource<'a>> {
    Ok(DenseWeightSource {
        input_norm: tensor(tensors, &bindings.input_norm.source)?,
        qkv,
        query_norm: optional_tensor(tensors, bindings.attention.query_norm)?,
        key_norm: optional_tensor(tensors, bindings.attention.key_norm)?,
        output,
        post_attention_norm: tensor(tensors, &bindings.post_attention_norm.source)?,
        sandwich: bindings
            .sandwich
            .map(|sandwich| {
                Ok::<_, Error>(DenseSandwichWeights {
                    attention_output_norm: tensor(tensors, &sandwich.attention_output_norm.source)?,
                    feed_forward_output_norm: tensor(
                        tensors,
                        &sandwich.feed_forward_output_norm.source,
                    )?,
                    layer_scalar: sandwich
                        .layer_scale
                        .map(|binding| tensor(tensors, &binding.source))
                        .transpose()?,
                })
            })
            .transpose()?,
        feed_forward,
    })
}

fn optional_tensor<'a>(
    tensors: &'a CudaTensorSet,
    binding: Option<&TensorBinding>,
) -> Result<Option<&'a CudaTensor>> {
    binding.map(|binding| tensor(tensors, &binding.source)).transpose()
}

pub(super) fn tensor<'a>(tensors: &'a CudaTensorSet, name: &str) -> Result<&'a CudaTensor> {
    tensors.get(name).ok_or_else(|| Error::MissingTensor(name.into()))
}
