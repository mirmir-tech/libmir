use std::time::{Duration, Instant};

use models::layout::{AttentionLayerType, DecoderConfig};

use super::layer::HybridLinearMoeLayer;
use crate::engine::{
    Array, DecoderCache, Error, ExpertFusionDecision, ModelTensors, NormWeight, QuantizedEmbedding,
    QuantizedLinear, Result, Stream, configure_expert_fusion, decode_graph,
};

#[derive(Debug)]
pub struct HybridLinearMoeModel {
    pub(super) layers: Vec<HybridLinearMoeLayer>,
    layer_types: Vec<AttentionLayerType>,
    cache_step: usize,
    pub(super) embedding: QuantizedEmbedding,
    pub(super) output: QuantizedLinear,
    pub(super) final_norm: NormWeight,
    pub(super) rms_norm_eps: f32,
    pub(super) hidden_size: usize,
    expert_fusion: ExpertFusionDecision,
}

impl HybridLinearMoeModel {
    pub fn load(
        tensors: &ModelTensors,
        decoder: &DecoderConfig,
        group_size: usize,
        cache_step: usize,
        stream: &Stream,
    ) -> Result<Self> {
        if !decoder.uses_hybrid_linear_moe_stack() || decoder.tie_word_embeddings {
            return Err(Error::InvalidModel(
                "hybrid linear MoE requires untied shared-expert decoder weights".into(),
            ));
        }
        let group_size = i32::try_from(group_size)?;
        let norm_shift = norm_shift(tensors, decoder)?;
        let mut layers = Vec::with_capacity(decoder.num_hidden_layers);
        for index in 0..decoder.num_hidden_layers {
            layers.push(HybridLinearMoeLayer::load(
                tensors, decoder, index, group_size, norm_shift, stream,
            )?);
        }
        let expert_fusion = configure_expert_fusion(
            &mut layers,
            stream,
            stream.config().fusion.shared_expert_gate_up,
        )?;
        Ok(Self {
            layers,
            layer_types: decoder.layer_types.clone(),
            cache_step,
            embedding: QuantizedEmbedding::load(
                tensors,
                "language_model.model.embed_tokens",
                group_size,
            )?,
            output: QuantizedLinear::load(tensors, "language_model.lm_head", group_size)?,
            final_norm: NormWeight::load_adjusted(
                tensors,
                "language_model.model.norm",
                norm_shift,
                stream,
            )?,
            rms_norm_eps: decoder.rms_norm_eps.to_string().parse()?,
            hidden_size: decoder.hidden_size,
            expert_fusion,
        })
    }

    pub fn new_cache(&self, stream: &Stream) -> Result<DecoderCache> {
        DecoderCache::new_hybrid_linear_with_format(
            &self.layer_types,
            self.cache_step,
            crate::engine::KvPageFormat::resolve(stream.config().kv_cache.dtype)?,
            stream.config().kv_cache.block_size,
        )
    }

    pub fn forward_decode(
        &self,
        token_ids: &Array,
        cache: &mut DecoderCache,
        position: i32,
        stream: &Stream,
    ) -> Result<Array> {
        let hidden = self.forward_hidden(token_ids, cache, position, false, stream)?;
        let logits = self.output.forward(&hidden, stream)?;
        decode_graph::export_once(&logits, stream)?;
        Ok(logits)
    }

    pub fn forward_prefill(
        &self,
        token_ids: &Array,
        cache: &mut DecoderCache,
        position: i32,
        stream: &Stream,
    ) -> Result<Array> {
        self.forward_hidden(token_ids, cache, position, true, stream)
    }

    #[must_use]
    pub fn fusion_summary(&self) -> (usize, usize, usize, usize) {
        let expert_gate_up =
            self.layers.iter().filter(|layer| layer.has_fused_expert_gate_up()).count();
        (0, 0, 0, expert_gate_up)
    }

    #[must_use]
    pub fn expert_fusion_summary(&self) -> String {
        self.expert_fusion.summary()
    }

    fn forward_hidden(
        &self,
        token_ids: &Array,
        cache: &mut DecoderCache,
        position: i32,
        causal: bool,
        stream: &Stream,
    ) -> Result<Array> {
        let hidden = self.embedding.lookup(token_ids, stream)?;
        self.forward_embedded(hidden, cache, position, causal, None, stream)
    }

    pub(super) fn forward_embedded(
        &self,
        mut hidden: Array,
        cache: &mut DecoderCache,
        position: i32,
        causal: bool,
        positions: Option<&Array>,
        stream: &Stream,
    ) -> Result<Array> {
        let sequence = hidden.shape()?.get(1).copied().unwrap_or_default();
        let profile = stream.config().diagnostics.profile_layers;
        let profile_graph = stream.config().diagnostics.profile_graph_build;
        let graph_started = Instant::now();
        let mut linear_graph = Duration::ZERO;
        let mut full_graph = Duration::ZERO;
        for (index, layer) in self.layers.iter().enumerate() {
            let started = Instant::now();
            hidden = layer
                .forward_with_positions(&hidden, cache, position, causal, positions, stream)?;
            if profile_graph {
                match layer.attention_kind() {
                    "gated_delta" => linear_graph += started.elapsed(),
                    _ => full_graph += started.elapsed(),
                }
            }
            if profile {
                hidden.async_eval()?;
                stream.synchronize()?;
                tracing::debug!(
                    layer = index,
                    attention = layer.attention_kind(),
                    milliseconds = started.elapsed().as_secs_f64() * 1_000.0,
                    "MLX hybrid linear MoE layer profile"
                );
            }
        }
        let output = self.final_norm.apply(&hidden, self.rms_norm_eps, stream)?;
        if profile_graph {
            tracing::debug!(
                sequence,
                total_ms = graph_started.elapsed().as_secs_f64() * 1_000.0,
                gated_delta_ms = linear_graph.as_secs_f64() * 1_000.0,
                gated_full_ms = full_graph.as_secs_f64() * 1_000.0,
                "MLX hybrid linear MoE graph-build profile"
            );
        }
        Ok(output)
    }
}

fn norm_shift(tensors: &ModelTensors, decoder: &DecoderConfig) -> Result<f32> {
    let index = decoder
        .layer_types
        .iter()
        .position(|layer| *layer == AttentionLayerType::Linear)
        .ok_or_else(|| Error::InvalidModel("missing linear attention layer".into()))?;
    let weight =
        tensors.get(&format!("language_model.model.layers.{index}.linear_attn.conv1d.weight"))?;
    let last_dimension = weight.shape()?.last().copied().unwrap_or_default();
    Ok(if last_dimension == 1 {
        0.0
    } else {
        1.0
    })
}
