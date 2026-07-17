use foundation::model::ModelManifest;
use models::{
    execution::{DecoderArchetype, ExecutionPlan},
    layout::{DecoderConfig, ModelLayout, ModelMetadata},
    tokenizer::{TextTokenizer, TokenizerInfo},
    weights::TensorCatalog,
};

use super::{KV_CACHE_STEP, LoadedModel, ModelInfo, prefill_step};
use crate::{
    MetalConfig, MetalProgressEvent,
    engine::{
        DecoderModel, ModelTensors, Stream, configure_recommended_wired_limit,
        dense_swiglu::DenseSwiGluModel, hybrid_linear_moe::HybridLinearMoeModel,
        hybrid_moe::HybridMoeModel, memory_stats,
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
        let plan = ExecutionPlan::discover(&decoder, &TensorCatalog::from_layout(&layout)?)?;
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
            prefixes: PrefixCache::new(prefix_cache_entries),
            sessions: std::collections::HashMap::new(),
        })
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
