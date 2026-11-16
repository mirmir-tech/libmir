use models::weights::{DenseDecoderLayerBindings, RoutedFeedForwardBindings};

use super::{common_source, tensor};
use crate::{
    CudaBackend, CudaTensorSet, DenseFeedForwardSource, DenseOutputSource, DenseQkvSource,
    DenseSwiGluLayerTemplate, Error, GgmlCheckpointWeight, ProjectionFormat, Result,
    RoutedExpertWeights,
};

/// A softmax-attention layer whose feed-forward is routed experts. Attention
/// takes the checkpoint's projection format; the experts their own storage.
pub(super) fn load(
    backend: &CudaBackend,
    block: crate::DenseSwiGluConfig,
    tensors: &CudaTensorSet,
    bindings: DenseDecoderLayerBindings<'_>,
    routed: RoutedFeedForwardBindings<'_>,
) -> Result<DenseSwiGluLayerTemplate> {
    let experts = RoutedExpertWeights::load(backend, tensors, routed, block)?;
    let feed_forward = DenseFeedForwardSource::Routed(&experts);
    let attention = bindings.attention;
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
            let weight = |binding| GgmlCheckpointWeight::load_binding(tensors, binding);
            let q = weight(attention.query)?;
            let k = weight(attention.key)?;
            let v = weight(attention.value)?;
            let output = weight(attention.output)?;
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
