mod latent;
mod state_space;
use architecture::cuda::CudaLayer;
pub(super) use latent::layers as latent_layers;
use models::{
    layout::{AttentionLayerType, DecoderConfig, LinearAttentionLayout},
    semantic::DecoderLayerSpec,
    weights::{HybridMixerBindings, MixedDecoderLayerBindings, TensorCatalog, WeightBindingPlan},
};
pub(super) use state_space::HybridLoad;

use super::SharedRoutedLayerTemplate;
use crate::{
    AffineGatedDeltaLayerConfig, AffineGatedDeltaLayerWeights, AffineGatedFullAttentionConfig,
    AffineGatedFullAttentionWeights, CudaAffineGatedDeltaLayer, CudaAffineGatedDeltaMoeLayer,
    CudaAffineGatedFullAttention, CudaAffineGatedFullAttentionMoeLayer, CudaBackend, CudaTensor,
    CudaTensorDType, CudaTensorSet, Error, Result,
    backend::feed_forward::{FeedForward, LayerNormConfig},
};

pub(super) fn build_layer(
    backend: &CudaBackend,
    decoder: &DecoderConfig,
    tensors: &CudaTensorSet,
    catalog: &TensorCatalog,
    (layer, spec, lowered): (usize, &DecoderLayerSpec, &CudaLayer),
    bindings: MixedDecoderLayerBindings<'_>,
    norm_shift: f32,
) -> Result<SharedRoutedLayerTemplate> {
    let feed_forward = crate::backend::feed_forward::load(
        backend,
        decoder.hidden_size,
        tensors,
        catalog,
        (&spec.feed_forward, &lowered.feed_forward),
        bindings.feed_forward,
    )?;
    let epsilon = decoder.rms_norm_eps.to_string().parse()?;
    match decoder.layer_type(layer) {
        AttentionLayerType::Linear => {
            let linear = decoder.linear_attention.as_ref().ok_or_else(|| {
                Error::UnsupportedDecoderLayer("missing parsed linear attention geometry".into())
            })?;
            let HybridMixerBindings::Linear(mixer_bindings) = bindings.mixer else {
                return Err(Error::UnsupportedDecoderLayer(
                    "linear layer has no linear mixer bindings".into(),
                ));
            };
            let mut weights = AffineGatedDeltaLayerWeights::load_bindings(tensors, mixer_bindings)?;
            let mixed_width = linear
                .key_heads
                .checked_mul(linear.key_head_dim)
                .and_then(|width| width.checked_mul(2))
                .and_then(|width| {
                    linear
                        .value_heads
                        .checked_mul(linear.value_head_dim)
                        .and_then(|value| width.checked_add(value))
                })
                .ok_or(Error::InvalidDecoderKernel("linear attention width overflow"))?;
            let (group_size, bits) = weights
                .qkv
                .affine_format(1, decoder.hidden_size, mixed_width)?
                .unwrap_or((0, 0));
            let attention = AffineGatedDeltaLayerConfig::from_linear_attention(
                decoder.hidden_size,
                linear,
                group_size,
                bits,
                decoder.rms_norm_eps,
                0.0,
            )?;
            if linear.layout == LinearAttentionLayout::LlamaCpp {
                weights.restore_llama_cpp_layout(backend, attention)?;
            }
            let config = LayerNormConfig {
                hidden_size: decoder.hidden_size,
                rms_norm_epsilon: epsilon,
                norm_weight_shift: norm_shift,
            };
            CudaAffineGatedDeltaMoeLayer::compose(
                backend,
                config,
                CudaAffineGatedDeltaLayer::new(backend, attention, weights)?,
                feed_forward,
                required_norm(tensors, &bindings.input_norm.source, decoder.hidden_size)?,
                required_norm(tensors, &bindings.post_attention_norm.source, decoder.hidden_size)?,
            )
            .map(|value| SharedRoutedLayerTemplate::Linear(Box::new(value)))
        },
        AttentionLayerType::Full => full_layer(
            backend, decoder, tensors, layer, bindings, feed_forward, epsilon, norm_shift,
        ),
        AttentionLayerType::Sliding => Err(Error::UnsupportedDecoderLayer(
            "shared-routed linear mixer stack contains sliding attention".into(),
        )),
        AttentionLayerType::StateSpace => Err(Error::UnsupportedDecoderLayer(
            "shared-routed linear mixer stack contains a state-space mixer".into(),
        )),
        AttentionLayerType::Latent => Err(Error::UnsupportedDecoderLayer(
            "shared-routed linear mixer stack contains latent attention".into(),
        )),
    }
}

#[allow(clippy::too_many_arguments)]
fn full_layer(
    backend: &CudaBackend,
    decoder: &DecoderConfig,
    tensors: &CudaTensorSet,
    layer: usize,
    bindings: MixedDecoderLayerBindings<'_>,
    feed_forward: FeedForward,
    epsilon: f32,
    norm_shift: f32,
) -> Result<SharedRoutedLayerTemplate> {
    let HybridMixerBindings::Softmax(mixer) = bindings.mixer else {
        return Err(Error::UnsupportedDecoderLayer(
            "full-attention layer has no softmax mixer bindings".into(),
        ));
    };
    let weights = AffineGatedFullAttentionWeights::load_bindings(tensors, mixer)?;
    let query = decoder
        .num_attention_heads
        .checked_mul(decoder.layer_head_dim(layer))
        .and_then(|width| width.checked_mul(2))
        .ok_or(Error::InvalidDecoderKernel("gated query width overflow"))?;
    let (group_size, bits) =
        weights.query.affine_format(1, decoder.hidden_size, query)?.unwrap_or((0, 0));
    let attention =
        AffineGatedFullAttentionConfig::from_decoder(decoder, layer, group_size, bits, norm_shift)?;
    let config = LayerNormConfig {
        hidden_size: decoder.hidden_size,
        rms_norm_epsilon: epsilon,
        norm_weight_shift: norm_shift,
    };
    CudaAffineGatedFullAttentionMoeLayer::compose(
        backend,
        config,
        CudaAffineGatedFullAttention::new(backend, attention, weights)?,
        feed_forward,
        required_norm(tensors, &bindings.input_norm.source, decoder.hidden_size)?,
        required_norm(tensors, &bindings.post_attention_norm.source, decoder.hidden_size)?,
    )
    .map(|value| SharedRoutedLayerTemplate::Full(Box::new(value)))
}

pub(super) fn infer_norm_shift(
    tensors: &CudaTensorSet,
    decoder: &DecoderConfig,
    bindings: &WeightBindingPlan,
) -> Result<f32> {
    let index = decoder
        .layer_types
        .iter()
        .position(|kind| *kind == AttentionLayerType::Linear)
        .ok_or_else(|| Error::UnsupportedDecoderLayer("missing linear attention layer".into()))?;
    let layer = bindings.mixed_decoder_layer(index)?;
    let HybridMixerBindings::Linear(linear) = layer.mixer else {
        return Err(Error::UnsupportedDecoderLayer(
            "linear layer has no linear mixer binding".into(),
        ));
    };
    // llama.cpp's GGUF conversion folds the unit offset into the norms.
    let folded = decoder
        .linear_attention
        .as_ref()
        .is_some_and(|linear| linear.layout == LinearAttentionLayout::LlamaCpp);
    if folded {
        return Ok(0.0);
    }
    let tensor = tensors
        .get(&linear.convolution.source)
        .ok_or_else(|| Error::MissingTensor(linear.convolution.source.clone()))?;
    Ok(if tensor.shape().last() == Some(&1) {
        0.0
    } else {
        1.0
    })
}

pub(super) fn required_norm(
    tensors: &CudaTensorSet,
    name: &str,
    hidden: usize,
) -> Result<CudaTensor> {
    let tensor = tensors.get(name).cloned().ok_or_else(|| Error::MissingTensor(name.into()))?;
    if tensor.shape() != [hidden] {
        return Err(Error::InvalidQuantizedTensor {
            name: name.into(),
            expected: vec![hidden],
            actual: tensor.shape().to_vec(),
        });
    }
    if tensor.dtype() != CudaTensorDType::Bf16 {
        return Err(Error::DTypeMismatch { name: name.into(), expected: "BF16" });
    }
    Ok(tensor)
}

/// Whether the runtime admission chose runs the hybrid or the latent layers
/// of the template, which builds every shared-routed runtime.
pub(super) fn composition(
    runtime: crate::CudaDecoderRuntime,
    max_sequence_blocks: usize,
) -> Result<(bool, bool)> {
    use crate::CudaDecoderRuntime;
    if max_sequence_blocks == 0 {
        return Err(Error::InvalidPagedKv("shared-routed sequence capacity is empty"));
    }
    match runtime {
        CudaDecoderRuntime::StateSpaceHybrid => Ok((true, false)),
        CudaDecoderRuntime::LatentAttention => Ok((false, true)),
        CudaDecoderRuntime::SharedRouted | CudaDecoderRuntime::DenseMixed => Ok((false, false)),
        CudaDecoderRuntime::Dense
        | CudaDecoderRuntime::DenseAndRouted
        | CudaDecoderRuntime::ClampedRouted => {
            Err(Error::State(format!("{runtime:?} is not a shared-routed runtime")))
        },
    }
}
