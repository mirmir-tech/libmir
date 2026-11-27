use models::{
    layout::DecoderConfig,
    semantic::SemanticModelSpec,
    weights::{DenseLayerFeedForwardBindings, TensorBinding, TensorCatalog},
};

use super::source::DenseLayerSource;
use crate::{
    CudaBackend, CudaTensor, CudaTensorSet, DenseDownSource, DenseFeedForwardSource,
    DenseGateUpSource, DenseOutputSource, DenseQkvSource, DenseSandwichWeights,
    DenseSwiGluLayerLoadConfig, DenseSwiGluLayerTemplate, DenseWeightSource, Error,
    PackedIntegerWeight, ProjectionFormat, Result,
    checkpoint::{dense::DenseLayerBindings, model::payload_bytes},
};

mod bf16;
mod bias;
mod direct_fp8;
mod ggml;
mod mxfp4;
mod mxfp8;
mod norm_bias;

pub(in crate::checkpoint) use norm_bias::append_bias as append_norm_bias;
mod nvfp4;
mod per_layer;
mod residual;
mod rope_rows;
mod routed;

impl CudaBackend {
    pub(crate) fn load_dense_swiglu_layer_tracked(
        &self,
        decoder: &DecoderConfig,
        (semantic, plan): (&SemanticModelSpec, &architecture::cuda::CudaDecoderPlan),
        catalog: &TensorCatalog,
        layer: usize,
        bindings: DenseLayerBindings<'_>,
        load: DenseSwiGluLayerLoadConfig,
    ) -> Result<(DenseSwiGluLayerTemplate, u64)> {
        let mut block = load.block(decoder, (semantic, plan), layer)?;
        let source = DenseLayerSource::discover(catalog, bindings)?;
        let mut tensors = source.upload(self, bindings, load.projection_format)?;
        // A layer without positions (Cohere 2's full-attention layers) has
        // no rotated rows to restore.
        if decoder.query_key_rows_interleaved
            && let Some(rotary) = block.attention.rotary
        {
            let head_dim = block.attention.cache.key_head_dim;
            rope_rows::restore(self, &mut tensors, bindings, (head_dim, rotary.dimensions))?;
        }
        let offset = semantic
            .decoder
            .layers
            .get(layer)
            .map_or(0.0, |spec| spec.input_norm.weight_offset);
        if offset != 0.0 {
            for norm in bindings.norms() {
                tensors.add_bf16(&norm.source, offset)?;
            }
        }
        norm_bias::pack(self, &mut tensors, bindings, &mut block)?;
        bias::require_applied(bindings, load.projection_format)?;
        let residual = residual::factor(semantic, decoder.attention_bias)?;
        if let Some(factor) = residual {
            residual::fold_dense(&tensors, bindings, factor)?;
        }
        let template = if let DenseLayerFeedForwardBindings::Gated(_) = bindings.feed_forward {
            self.load_gated(block, &tensors, bindings, (load.projection_format, residual))?
        } else {
            routed::load(self, block, &tensors, bindings, residual)?
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
        bindings: DenseLayerBindings<'_>,
        (format, residual): (ProjectionFormat, Option<f32>),
    ) -> Result<DenseSwiGluLayerTemplate> {
        Ok(match format {
            ProjectionFormat::Affine => load_affine(self, block, tensors, bindings)?,
            ProjectionFormat::Bf16 => bf16::load(self, block, tensors, bindings)?,
            ProjectionFormat::DirectFp8 => direct_fp8::load(self, block, tensors, bindings)?,
            ProjectionFormat::Ggml => ggml::load(self, block, tensors, bindings, residual)?,
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
    bindings: DenseLayerBindings<'_>,
) -> Result<DenseSwiGluLayerTemplate> {
    let feed_forward = bindings.feed_forward.gated()?;
    let attention = bindings.attention.projected()?;
    let affine = |binding| crate::AffineQuantizedWeight::load_binding(tensors, binding);
    let q = affine(attention.query)?;
    let k = affine(attention.key)?;
    let v = affine(attention.value)?;
    let output = affine(attention.output)?;
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
    bindings: DenseLayerBindings<'_>,
) -> Result<DenseSwiGluLayerTemplate> {
    let feed_forward = bindings.feed_forward.gated()?;
    let hidden = block.attention.hidden_size;
    let head = block.attention.cache.key_head_dim;
    let query = block.attention.query_heads * head;
    let key_value = block.attention.cache.kv_heads * head;
    let intermediate = block.intermediate_size;
    let attention = bindings.attention.projected()?;
    let q = packed_integer(tensors, attention.query, hidden, query)?;
    let k = packed_integer(tensors, attention.key, hidden, key_value)?;
    let v = packed_integer(tensors, attention.value, hidden, key_value)?;
    let output = packed_integer(tensors, attention.output, query, hidden)?;
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

pub(super) fn common_source<'a>(
    tensors: &'a CudaTensorSet,
    bindings: DenseLayerBindings<'_>,
    qkv: DenseQkvSource<'a>,
    output: DenseOutputSource<'a>,
    feed_forward: DenseFeedForwardSource<'a>,
) -> Result<DenseWeightSource<'a>> {
    Ok(DenseWeightSource {
        input_norm: optional_tensor(tensors, bindings.input_norm)?,
        qkv,
        query_norm: optional_tensor(tensors, bindings.attention.query_norm())?,
        key_norm: optional_tensor(tensors, bindings.attention.key_norm())?,
        output,
        post_attention_norm: optional_tensor(tensors, bindings.post_attention_norm)?,
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
        per_layer_input: bindings
            .per_layer_input
            .map(|input| per_layer::load(tensors, input))
            .transpose()?,
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
