use std::collections::BTreeMap;

use models::tokenizer::TextTokenizer;
use runtime::{
    backend::BackendInfo,
    kv::CacheConfig,
    trace::{
        ModelTrace, TraceDTypeCount, TraceFiniteValidation, TraceModel, TraceTensors,
        TraceTokenizer,
    },
};

use super::model::LoadedModel;
use crate::Result;

mod description;
mod execution;
mod kv;
use description::{attention_counts, execution as execution_trace, weights as weight_trace};
use execution::{acceleration, actions, mlp_layout, warnings};

pub(super) fn build(
    model: &LoadedModel,
    backend: BackendInfo,
    cache: CacheConfig,
) -> Result<ModelTrace> {
    let (full_attention_layers, sliding_attention_layers) = attention_counts(model);
    let sessions = model.sessions()?.len();
    Ok(ModelTrace {
        model: TraceModel {
            id: model.manifest.id.clone(),
            root: model.layout.root.display().to_string(),
            model_type: model.metadata.model_type.clone(),
            dtype: model.metadata.dtype.clone(),
            architectures: model.metadata.architectures.clone(),
            context_len: model.metadata.context_len,
            quantization: model.metadata.quantization.clone(),
            quantization_group_size: model.metadata.quantization_group_size,
            quantization_mode: model.metadata.quantization_mode.clone(),
        },
        backend,
        acceleration: acceleration(model),
        decoder: execution_trace(
            model.decoder.as_ref(),
            model.encoder.as_ref(),
            full_attention_layers,
            sliding_attention_layers,
        ),
        tokenizer: tokenizer_trace(model),
        tensors: tensor_trace(model),
        weights: weight_trace(model),
        kv_cache: kv::build(model, cache, sessions),
        actions: actions(model, cache),
        warnings: warnings(model),
    })
}

fn tokenizer_trace(model: &LoadedModel) -> TraceTokenizer {
    match TextTokenizer::from_layout(&model.layout) {
        Ok(tokenizer) => {
            let info = tokenizer.info();
            TraceTokenizer {
                path: Some(info.path.display().to_string()),
                kind: Some(format!("{:?}", info.kind)),
                vocab_size: Some(info.vocab_size),
                stop_token_ids: info.stop_token_ids,
                error: None,
            }
        },
        Err(error) => TraceTokenizer {
            path: model.layout.tokenizer_path.as_ref().map(|path| path.display().to_string()),
            kind: None,
            vocab_size: None,
            stop_token_ids: Vec::new(),
            error: Some(error.to_string()),
        },
    }
}

fn tensor_trace(model: &LoadedModel) -> TraceTensors {
    let mut dtypes = BTreeMap::<String, (usize, usize)>::new();
    for tensor in &model.catalog.tensors {
        let entry = dtypes.entry(tensor.dtype.clone()).or_default();
        entry.0 += 1;
        entry.1 += tensor.shape.iter().product::<usize>();
    }
    let readiness = model.vision_readiness.as_ref().map_or_else(
        || format!("native CUDA {:?} task loaded", model.task_plan.task()),
        |vision| {
            format!(
                "native CUDA {:?} text model loaded; {:?} vision discovered; {}",
                model.task_plan.task(),
                model.vision.as_ref().map(models::layout::VisionConfig::pipeline),
                vision.summary()
            )
        },
    );
    TraceTensors {
        tensor_count: model.catalog.len(),
        native_tensor_count: model.catalog.len(),
        weight_files: model.layout.weights.len(),
        native_shards: model.layout.weights.len(),
        weight_bytes: model.layout.weights.iter().map(|weight| weight.bytes).sum(),
        tokenizer: model.layout.has_tokenizer(),
        safetensors_index: model.layout.safetensors_index_path.is_some(),
        readiness,
        missing: model
            .vision_readiness
            .as_ref()
            .map(|vision| vision.missing.clone())
            .unwrap_or_default(),
        native_dtypes: dtypes
            .into_iter()
            .map(|(dtype, (tensors, elements))| TraceDTypeCount { dtype, tensors, elements })
            .collect(),
        finite_validation: TraceFiniteValidation {
            mode: "checkpoint metadata".into(),
            checked_tensors: 0,
            checked_elements: 0,
        },
    }
}
