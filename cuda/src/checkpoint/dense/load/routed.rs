use models::weights::DenseLayerFeedForwardBindings;

use super::{common_source, tensor};
use crate::{
    CudaBackend, CudaTensorSet, DenseFeedForwardSource, DenseOutputSource, DenseQkvSource,
    DenseSwiGluLayerTemplate, Error, GgmlCheckpointWeight, ProjectionFormat, Result,
    RoutedExpertWeights, checkpoint::dense::DenseLayerBindings,
};

/// A softmax-attention layer whose feed-forward is routed experts, with or
/// without a shared expert. Attention takes the checkpoint's projection
/// format; the experts their own storage.
pub(super) fn load(
    backend: &CudaBackend,
    block: crate::DenseSwiGluConfig,
    tensors: &CudaTensorSet,
    bindings: DenseLayerBindings<'_>,
    residual: Option<f32>,
) -> Result<DenseSwiGluLayerTemplate> {
    let mut experts = match bindings.feed_forward {
        DenseLayerFeedForwardBindings::Routed(routed) => {
            RoutedExpertWeights::load(backend, tensors, routed, block)?
        },
        DenseLayerFeedForwardBindings::SharedRouted(shared) => {
            RoutedExpertWeights::load_shared(backend, tensors, shared, block)?
        },
        DenseLayerFeedForwardBindings::Gated(_) => {
            return Err(Error::InvalidExecutionPlan("a gated MLP layer has no routed experts"));
        },
    };
    // The residual multiplier ends both branches; BF16 attention outputs
    // were scaled at upload.
    if let Some(factor) = residual {
        experts.scale_down(backend, factor)?;
    }
    let feed_forward = DenseFeedForwardSource::Routed(&experts);
    let attention = bindings.attention.projected()?;
    match block.attention.projection_format {
        ProjectionFormat::Bf16 => {
            let qkv = backend.pack_bf16_linears([
                tensor(tensors, &attention.query.source)?,
                tensor(tensors, &attention.key.source)?,
                tensor(tensors, &attention.value.source)?,
            ])?;
            let output = DenseOutputSource::Bf16(tensor(tensors, &attention.output.source)?);
            backend.prepare_dense_swiglu_layer_template(
                block,
                common_source(tensors, bindings, DenseQkvSource::Bf16(&qkv), output, feed_forward)?,
            )
        },
        ProjectionFormat::Ggml => {
            let weight = |binding| GgmlCheckpointWeight::load_binding_with_bias(tensors, binding);
            let q = weight(attention.query)?;
            let k = weight(attention.key)?;
            let v = weight(attention.value)?;
            let output = match residual {
                Some(factor) => weight(attention.output)?.scaled(backend, factor)?,
                None => weight(attention.output)?,
            };
            backend.prepare_dense_swiglu_layer_template(
                block,
                common_source(
                    tensors,
                    bindings,
                    DenseQkvSource::Ggml([&q, &k, &v]),
                    DenseOutputSource::Ggml(&output),
                    feed_forward,
                )?,
            )
        },
        format => Err(Error::UnsupportedDecoderLayer(format!(
            "CUDA routed-expert layers have no {format:?} attention loader"
        ))),
    }
}
