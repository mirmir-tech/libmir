use models::{
    semantic::{KeyValueRelation, MixerSpec, QkNormalization},
    weights::TensorStorage,
};
use runtime::trace::{TraceTensors, TraceWeights};

use crate::{
    engine::lowering::{self, DecoderLowering, DecoderRuntime},
    native::model::LoadedModel,
};

pub(super) fn tensors(model: &LoadedModel) -> TraceTensors {
    let info = &model.info;
    let readiness = info.vision_readiness.as_ref().map_or_else(
        || format!("native {:?} task execution plan loaded", info.task_plan.task()),
        |vision| {
            format!(
                "native {:?} text execution plan loaded; {:?} discovered; {}",
                info.task_plan.task(),
                info.vision.as_ref().map(models::layout::VisionConfig::pipeline),
                vision.summary()
            )
        },
    );
    TraceTensors {
        tensor_count: info.tensor_count,
        native_tensor_count: info.tensor_count,
        weight_files: info.layout.weights.len(),
        native_shards: info.layout.weights.len(),
        weight_bytes: info.weight_bytes,
        tokenizer: info.layout.has_tokenizer(),
        safetensors_index: info.layout.safetensors_index_path.is_some(),
        readiness,
        missing: info
            .vision_readiness
            .as_ref()
            .map(|vision| vision.missing.clone())
            .unwrap_or_default(),
        native_dtypes: Vec::new(),
        finite_validation: runtime::trace::TraceFiniteValidation {
            mode: "deferred to MLX kernel execution".into(),
            checked_tensors: 0,
            checked_elements: 0,
        },
    }
}

pub(super) fn weights(model: &LoadedModel) -> TraceWeights {
    let info = &model.info;
    let Some(contract) = info.contract.as_ref() else {
        return info.encoder.as_ref().map_or_else(unknown_weights, |encoder| TraceWeights {
            token_embeddings: "new.embeddings.word_embeddings".into(),
            final_norm: "post-attention and post-MLP LayerNorm per encoder layer".into(),
            output_head: "new.pooler.dense -> tanh -> classifier".into(),
            output_tied: false,
            layer_count: encoder.num_hidden_layers,
            attention_layout: "packed QKV with bidirectional self-attention".into(),
            mlp_layout: "packed gated exact-GELU feed-forward".into(),
            linear_bias_count: encoder.num_hidden_layers.saturating_mul(4).saturating_add(2),
        });
    };
    let spec = &contract.semantic;
    let Ok(boundary) = contract
        .bindings
        .decoder_boundary_with_tied_output(spec.decoder.tie_word_embeddings)
    else {
        return unknown_weights();
    };
    let Ok(lowering) = lowering::plan(spec) else {
        return unknown_weights();
    };
    TraceWeights {
        token_embeddings: boundary.embedding.source.clone(),
        final_norm: boundary.final_norm.source.clone(),
        output_head: boundary.output.source.clone(),
        output_tied: spec.decoder.tie_word_embeddings,
        layer_count: spec.decoder.layers.len(),
        attention_layout: attention_layout(&lowering, spec).into(),
        mlp_layout: mlp_layout(&lowering).into(),
        linear_bias_count: contract
            .bindings
            .tensors
            .iter()
            .map(has_bias)
            .filter(|biased| *biased)
            .count(),
    }
}

fn attention_layout(
    lowering: &DecoderLowering,
    spec: &models::semantic::SemanticModelSpec,
) -> &'static str {
    match lowering.runtime() {
        DecoderRuntime::ClampedRouted => {
            "biased grouped-query attention with learned sinks and semantic windowing"
        },
        DecoderRuntime::DenseAndRouted => "RMS-normalized attention with shared K/V semantics",
        DecoderRuntime::SharedRouted => {
            "linear recurrence plus gated RMS-normalized grouped-query attention"
        },
        DecoderRuntime::Dense => dense_attention_layout(spec),
    }
}

fn dense_attention_layout(spec: &models::semantic::SemanticModelSpec) -> &'static str {
    let attentions = spec.decoder.layers.iter().filter_map(|layer| match &layer.mixer {
        MixerSpec::SoftmaxAttention(attention) => Some(attention),
        MixerSpec::LinearAttention(_) => None,
    });
    if attentions.clone().all(|attention| attention.sinks) {
        "grouped-query attention with learned sinks"
    } else if attentions.clone().all(|attention| {
        attention.qk_normalization == QkNormalization::QueryKeyRms
            && attention.key_value_relation == KeyValueRelation::Separate
    }) {
        "RMS-normalized grouped-query attention"
    } else {
        "grouped-query attention"
    }
}

fn mlp_layout(lowering: &DecoderLowering) -> &'static str {
    match lowering.runtime() {
        DecoderRuntime::Dense => "dense gated feed-forward",
        DecoderRuntime::DenseAndRouted => "dense GELU plus routed experts",
        DecoderRuntime::SharedRouted => "shared expert plus routed experts",
        DecoderRuntime::ClampedRouted => "clamped routed SwiGLU experts",
    }
}

fn has_bias(binding: &models::weights::TensorBinding) -> bool {
    match &binding.storage {
        TensorStorage::Dense { bias, .. }
        | TensorStorage::Float8 { bias, .. }
        | TensorStorage::BlockQuantized { bias, .. } => bias.is_some(),
        TensorStorage::AffineQuantized { output_bias, .. } => output_bias.is_some(),
        TensorStorage::PackedInt8 { .. }
        | TensorStorage::PackedInt4 { .. }
        | TensorStorage::Awq { .. }
        | TensorStorage::Gptq { .. }
        | TensorStorage::BitsAndBytes4Bit { .. }
        | TensorStorage::Auxiliary { .. } => false,
    }
}

fn unknown_weights() -> TraceWeights {
    TraceWeights {
        token_embeddings: "unknown".into(),
        final_norm: "unknown".into(),
        output_head: "unknown".into(),
        output_tied: false,
        layer_count: 0,
        attention_layout: "unknown".into(),
        mlp_layout: "unknown".into(),
        linear_bias_count: 0,
    }
}
