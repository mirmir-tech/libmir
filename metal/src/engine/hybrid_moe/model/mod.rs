use std::time::Instant;

use models::layout::DecoderConfig;

use super::{
    HybridMoeLayer, HybridMoeLayerConfig,
    layer::{fused_attention_enabled, fused_gate_up_enabled, profile_components},
};
use crate::engine::{
    Array, DecoderCache, ExpertFusionDecision, ModelTensors, QuantizedEmbedding, Result, Stream,
    configure_expert_fusion, decode_graph,
};

#[derive(Debug)]
pub struct HybridMoeModel {
    pub(super) layers: Vec<HybridMoeLayer>,
    cache_windows: Vec<Option<usize>>,
    cache_step: usize,
    pub(super) embedding: QuantizedEmbedding,
    pub(super) final_norm: Array,
    pub(super) embed_scale: f32,
    pub(super) softcap: Option<f32>,
    expert_fusion: ExpertFusionDecision,
}

impl HybridMoeModel {
    pub fn load(
        tensors: &ModelTensors,
        decoder: &DecoderConfig,
        group_size: usize,
        cache_step: usize,
        stream: &Stream,
    ) -> Result<Self> {
        if !decoder.uses_hybrid_routed_moe_stack() {
            return Err(crate::engine::Error::InvalidModel(
                "native hybrid MoE model requires compatible routed-MoE decoder features".into(),
            ));
        }
        let embedding = QuantizedEmbedding::load(
            tensors,
            "language_model.model.embed_tokens",
            i32::try_from(group_size)?,
        )?;
        let mut layers = Vec::with_capacity(decoder.num_hidden_layers);
        let mut cache_windows = Vec::with_capacity(decoder.num_hidden_layers);
        for index in 0..decoder.num_hidden_layers {
            let config = HybridMoeLayerConfig::from_decoder(index, decoder, group_size)?;
            layers.push(HybridMoeLayer::load(tensors, config, stream)?);
            cache_windows.push(config.max_context);
        }
        if fused_attention_enabled(stream) || fused_gate_up_enabled(stream) {
            for layer in &layers {
                layer.warm_fused_projections()?;
            }
            stream.synchronize()?;
        }
        let expert_fusion = configure_expert_fusion(
            &mut layers,
            stream,
            stream.config().fusion.routed_expert_gate_up,
        )?;
        let embed_scale = decoder.hidden_size.to_string().parse::<f32>()?.sqrt();
        Ok(Self {
            layers,
            cache_windows,
            cache_step,
            embedding,
            final_norm: tensors.get("language_model.model.norm.weight")?,
            embed_scale,
            softcap: decoder
                .final_logit_softcapping
                .map(|value| value.to_string().parse())
                .transpose()?,
            expert_fusion,
        })
    }

    pub fn new_cache(&self) -> Result<DecoderCache> {
        DecoderCache::new(&self.cache_windows, self.cache_step)
    }

    pub fn forward_decode(
        &self,
        token_ids: &Array,
        cache: &mut DecoderCache,
        position: i32,
        stream: &Stream,
    ) -> Result<Array> {
        self.forward_decode_with_softcap(token_ids, cache, position, true, stream)
    }

    pub fn forward_greedy_decode(
        &self,
        token_ids: &Array,
        cache: &mut DecoderCache,
        position: i32,
        stream: &Stream,
    ) -> Result<Array> {
        self.forward_decode_with_softcap(token_ids, cache, position, false, stream)
    }

    fn forward_decode_with_softcap(
        &self,
        token_ids: &Array,
        cache: &mut DecoderCache,
        position: i32,
        apply_softcap: bool,
        stream: &Stream,
    ) -> Result<Array> {
        let hidden = self.forward_hidden(token_ids, cache, position, false, stream)?;
        let profile_components = profile_components(stream);
        let logits_started = Instant::now();
        let logits = self.logits(&hidden, apply_softcap, stream)?;
        decode_graph::export_once(&logits, stream)?;
        if profile_components {
            logits.async_eval()?;
            stream.synchronize()?;
            tracing::debug!(
                component = "logits",
                milliseconds = logits_started.elapsed().as_secs_f64() * 1_000.0,
                "MLX hybrid MoE component profile"
            );
        }
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

    fn forward_hidden(
        &self,
        token_ids: &Array,
        cache: &mut DecoderCache,
        position: i32,
        causal: bool,
        stream: &Stream,
    ) -> Result<Array> {
        let profile_components = !causal && profile_components(stream);
        let embedding_started = Instant::now();
        let mut hidden = self.embedding.lookup(token_ids, stream)?;
        hidden = hidden.multiply_scalar(self.embed_scale, stream)?;
        if profile_components {
            hidden.async_eval()?;
            stream.synchronize()?;
            tracing::debug!(
                component = "embedding",
                milliseconds = embedding_started.elapsed().as_secs_f64() * 1_000.0,
                "MLX hybrid MoE component profile"
            );
        }
        let profile = stream.config().diagnostics.profile_layers;
        let evaluation_step = causal
            .then_some(stream.config().diagnostics.prefill_evaluation_layers)
            .flatten();
        let layer_count = self.layers.len();
        for (index, (layer, cache)) in
            self.layers.iter().zip(cache.attention_caches_mut()?.iter_mut()).enumerate()
        {
            let started = Instant::now();
            hidden = layer.forward_decode(&hidden, Some(cache), position, causal, stream)?;
            let evaluate = profile
                || evaluation_step
                    .is_some_and(|step| (index + 1) % step == 0 || index + 1 == layer_count);
            if evaluate {
                hidden.async_eval()?;
                stream.synchronize()?;
            }
            if profile {
                tracing::debug!(
                    layer = index,
                    milliseconds = started.elapsed().as_secs_f64() * 1_000.0,
                    "MLX hybrid MoE layer profile"
                );
            }
        }
        Ok(hidden)
    }

    fn logits(&self, hidden: &Array, apply_softcap: bool, stream: &Stream) -> Result<Array> {
        let hidden = hidden.rms_norm(&self.final_norm, 1.0e-6, stream)?;
        let logits = self.embedding.project(&hidden, stream)?;
        match (self.softcap, apply_softcap) {
            (Some(cap), true) => logits.logit_softcap(cap, stream),
            _ => Ok(logits),
        }
    }

    #[must_use]
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    #[must_use]
    pub fn fusion_summary(&self) -> (usize, usize, usize, usize) {
        self.layers.iter().fold((0, 0, 0, 0), |counts, layer| {
            let (attention, key_value, gate_up, expert_gate_up) = layer.fusion_summary();
            (
                counts.0 + usize::from(attention),
                counts.1 + usize::from(key_value),
                counts.2 + usize::from(gate_up),
                counts.3 + usize::from(expert_gate_up),
            )
        })
    }

    #[must_use]
    pub fn expert_fusion_summary(&self) -> String {
        self.expert_fusion.summary()
    }
}
