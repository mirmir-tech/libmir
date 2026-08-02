use models::{
    layout::{DecoderConfig, VisionConfig},
    weights::WeightBindingPlan,
};

use super::super::{KV_CACHE_STEP, LoadedVisionModel};
use crate::{
    engine::{
        DecoderModel, KvPageFormat, ModelTensors, PooledVisionTower, SpatialMergeVisionTower,
        Stream,
        clamped_routed::ClampedRoutedModel,
        dense::swiglu::DenseSwiGluModel,
        hybrid_linear_moe::HybridLinearMoeModel,
        hybrid_moe::HybridMoeModel,
        lowering::{DecoderLowering, DecoderRuntime, MixerLowering},
    },
    native::error::{Error, Result},
};

pub(super) fn validate_kv_storage(
    decoder: &DecoderConfig,
    lowering: &DecoderLowering,
    vision: Option<&VisionConfig>,
    dtype: runtime::kv::KvCacheDType,
) -> Result<()> {
    let format = KvPageFormat::resolve(dtype)?;
    if !format.quantized() {
        return Ok(());
    }
    if vision.is_some() {
        return Err(Error::UnsupportedModel(
            "INT8 Metal K/V does not yet support multimodal prefix masks".into(),
        ));
    }
    if lowering
        .layers()
        .iter()
        .any(|layer| matches!(layer.mixer, MixerLowering::Softmax { window: Some(_), .. }))
    {
        return Err(Error::UnsupportedModel(
            "INT8 Metal K/V does not yet support sliding-window layers".into(),
        ));
    }
    for layer in lowering.layers() {
        let index = layer.index;
        if matches!(layer.mixer, MixerLowering::Softmax { .. })
            && (!decoder.layer_head_dim(index).is_multiple_of(4)
                || decoder.layer_head_dim(index) > 512)
        {
            return Err(Error::UnsupportedModel(format!(
                "INT8 Metal K/V requires attention head dimensions divisible by 4 and at most 512; layer {index} has {}",
                decoder.layer_head_dim(index)
            )));
        }
    }
    Ok(())
}

pub(super) fn load_vision_model(
    config: Option<&VisionConfig>,
    tensors_ready: bool,
    bindings: Option<&WeightBindingPlan>,
    tensors: &ModelTensors,
    stream: &Stream,
) -> Result<Option<LoadedVisionModel>> {
    if !tensors_ready {
        return Ok(None);
    }
    match config {
        Some(VisionConfig::PooledEncoder(config)) => {
            let projection = bindings
                .and_then(|plan| {
                    plan.binding(&models::weights::LogicalTensorRole::VisionProjection)
                })
                .ok_or_else(|| {
                    Error::UnsupportedModel("pooled vision projection binding is missing".into())
                })?;
            Ok(Some(LoadedVisionModel::PooledEncoder(PooledVisionTower::load(
                tensors, config, projection, stream,
            )?)))
        },
        Some(VisionConfig::SpatialMergeEncoder(config)) => {
            Ok(Some(LoadedVisionModel::SpatialMergeEncoder(SpatialMergeVisionTower::load(
                tensors, config, stream,
            )?)))
        },
        None => Ok(None),
    }
}

pub(super) fn load_decoder_model(
    lowering: &DecoderLowering,
    tensors: &ModelTensors,
    decoder: &DecoderConfig,
    bindings: &WeightBindingPlan,
    stream: &Stream,
) -> Result<DecoderModel> {
    if lowering.layers().len() != decoder.num_hidden_layers {
        return Err(Error::UnsupportedModel(format!(
            "Metal lowered {} layers for a {}-layer decoder",
            lowering.layers().len(),
            decoder.num_hidden_layers
        )));
    }
    match lowering.runtime() {
        DecoderRuntime::ClampedRouted => Ok(DecoderModel::new(ClampedRoutedModel::load(
            tensors,
            decoder,
            bindings,
            lowering.layers(),
            KV_CACHE_STEP,
            stream,
        )?)),
        DecoderRuntime::DenseAndRouted => {
            load_dense_routed(lowering, tensors, decoder, bindings, stream)
        },
        DecoderRuntime::Dense => load_dense(lowering, tensors, decoder, bindings, stream),
        DecoderRuntime::SharedRouted => Ok(DecoderModel::new(HybridLinearMoeModel::load(
            tensors,
            decoder,
            bindings,
            lowering.layers(),
            KV_CACHE_STEP,
            stream,
        )?)),
    }
}

fn load_dense_routed(
    lowering: &DecoderLowering,
    tensors: &ModelTensors,
    decoder: &DecoderConfig,
    bindings: &WeightBindingPlan,
    stream: &Stream,
) -> Result<DecoderModel> {
    Ok(DecoderModel::new(HybridMoeModel::load_bindings(
        tensors,
        decoder,
        bindings,
        lowering.layers(),
        KV_CACHE_STEP,
        stream,
    )?))
}

fn load_dense(
    lowering: &DecoderLowering,
    tensors: &ModelTensors,
    decoder: &DecoderConfig,
    bindings: &WeightBindingPlan,
    stream: &Stream,
) -> Result<DecoderModel> {
    Ok(DecoderModel::new(DenseSwiGluModel::load(
        tensors,
        decoder,
        bindings,
        lowering.layers(),
        KV_CACHE_STEP,
        stream,
    )?))
}
