use foundation::model::ModelManifest;
use models::{
    execution::{DecoderArchetype, ExecutionPlan},
    layout::{AttentionLayerType, DecoderConfig, ModelLayout, ModelMetadata, VisionConfig},
    tokenizer::{TextTokenizer, TokenizerInfo},
    weights::{TensorCatalog, TensorReadiness, VisionTensorSchema},
};

use super::{KV_CACHE_STEP, LoadedModel, LoadedVisionModel, ModelInfo, prefill_step};
use crate::{
    MetalConfig, MetalProgressEvent,
    engine::{
        DecoderModel, KvPageFormat, ModelTensors, PooledVisionTower, SpatialMergeVisionTower,
        Stream, configure_recommended_wired_limit, dense_swiglu::DenseSwiGluModel,
        hybrid_linear_moe::HybridLinearMoeModel, hybrid_moe::HybridMoeModel, memory_stats,
    },
    native::{
        error::{Error, Result},
        prefix::PrefixCache,
    },
};

impl LoadedModel {
    #[cfg(test)]
    pub fn load(
        manifest: &ModelManifest,
        progress: &mut dyn FnMut(MetalProgressEvent),
    ) -> Result<Self> {
        Self::load_with_config(manifest, Arc::default(), progress)
    }

    pub fn load_with_config(
        manifest: &ModelManifest,
        config: Arc<MetalConfig>,
        progress: &mut dyn FnMut(MetalProgressEvent),
    ) -> Result<Self> {
        let layout = ModelLayout::inspect(&manifest.path)?;
        let metadata = ModelMetadata::from_layout(&layout)?;
        let decoder = DecoderConfig::from_layout(&layout)?;
        let catalog = TensorCatalog::from_layout(&layout)?;
        let plan = ExecutionPlan::discover(&decoder, &catalog)?;
        let vision = VisionConfig::from_layout(&layout)?;
        validate_kv_storage(&decoder, vision.as_ref(), config.kv_cache.dtype)?;
        let vision_readiness = vision
            .as_ref()
            .map(|config| VisionTensorSchema::discover(config).readiness(&catalog));
        if !plan.is_native_implemented() {
            return Err(Error::UnsupportedModel(
                "native execution path for the detected hybrid linear MoE decoder is pending"
                    .into(),
            ));
        }
        let group_size = metadata.quantization_group_size.ok_or_else(|| {
            Error::UnsupportedModel("native quantized weights require group_size".into())
        })?;
        let load_stream = Stream::new_cpu()?;
        let tensors = ModelTensors::load_layout_materialized_with_progress(
            &layout,
            &load_stream,
            |loaded, total, detail| {
                progress(MetalProgressEvent::load_weights(loaded, total, detail));
            },
        )?;
        let tensor_count = tensors.len();
        let stream = Stream::new_gpu_with_config(config)?;
        let _configured_wired_limit = configure_recommended_wired_limit()?;
        let model = load_decoder_model(plan.decoder, &tensors, &decoder, group_size, &stream)?;
        let vision_model = load_vision_model(
            vision.as_ref(),
            vision_readiness.as_ref().is_some_and(TensorReadiness::is_ready),
            &tensors,
            &stream,
        )?;
        let metal_memory = memory_stats()?;
        let (tokenizer, tokenizer_error) = tokenizer_info(&layout);
        let weight_bytes = layout.weights.iter().map(|weight| weight.bytes).sum();
        let prefix_cache_entries = stream.config().cache.prefix_cache_entries;
        Ok(Self {
            info: ModelInfo {
                manifest: manifest.clone(),
                layout,
                metadata,
                decoder,
                vision,
                vision_readiness,
                plan,
                tensor_count,
                weight_bytes,
                cache_step: KV_CACHE_STEP,
                prefill_step: prefill_step(plan.decoder, stream.config().cache.prefill_step),
                tokenizer,
                tokenizer_error,
                metal_memory,
            },
            stream,
            model,
            vision_model,
            prefixes: PrefixCache::new(prefix_cache_entries),
            sessions: std::collections::HashMap::new(),
        })
    }
}

fn validate_kv_storage(
    decoder: &DecoderConfig,
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
    if decoder.layer_types.contains(&AttentionLayerType::Sliding) {
        return Err(Error::UnsupportedModel(
            "INT8 Metal K/V does not yet support sliding-window layers".into(),
        ));
    }
    for (index, layer) in decoder.layer_types.iter().enumerate() {
        if *layer != AttentionLayerType::Linear
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

fn load_vision_model(
    config: Option<&VisionConfig>,
    tensors_ready: bool,
    tensors: &ModelTensors,
    stream: &Stream,
) -> Result<Option<LoadedVisionModel>> {
    if !tensors_ready {
        return Ok(None);
    }
    match config {
        Some(VisionConfig::PooledEncoder(config)) => {
            PooledVisionTower::load(tensors, config, stream)
                .map(LoadedVisionModel::PooledEncoder)
                .map(Some)
                .map_err(Into::into)
        },
        Some(VisionConfig::SpatialMergeEncoder(config)) => {
            SpatialMergeVisionTower::load(tensors, config, stream)
                .map(LoadedVisionModel::SpatialMergeEncoder)
                .map(Some)
                .map_err(Into::into)
        },
        None => Ok(None),
    }
}

fn load_decoder_model(
    archetype: DecoderArchetype,
    tensors: &ModelTensors,
    decoder: &DecoderConfig,
    group_size: usize,
    stream: &Stream,
) -> Result<DecoderModel> {
    match archetype {
        DecoderArchetype::HybridMoe => Ok(DecoderModel::HybridMoe(HybridMoeModel::load(
            tensors, decoder, group_size, KV_CACHE_STEP, stream,
        )?)),
        DecoderArchetype::DenseSwiGlu => Ok(DecoderModel::DenseSwiGlu(DenseSwiGluModel::load(
            tensors, decoder, group_size, KV_CACHE_STEP, stream,
        )?)),
        DecoderArchetype::HybridLinearMoe => Ok(DecoderModel::HybridLinearMoe(
            HybridLinearMoeModel::load(tensors, decoder, group_size, KV_CACHE_STEP, stream)?,
        )),
    }
}

fn tokenizer_info(layout: &ModelLayout) -> (Option<TokenizerInfo>, Option<String>) {
    TextTokenizer::from_layout(layout).map_or_else(
        |error| (None, Some(error.to_string())),
        |tokenizer| (Some(tokenizer.info()), None),
    )
}
use std::sync::Arc;
