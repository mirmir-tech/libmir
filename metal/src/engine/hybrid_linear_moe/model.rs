use models::{
    layout::DecoderConfig,
    weights::{HybridMixerBindings, WeightBindingPlan},
};

use super::layer::HybridLinearMoeLayer;
use crate::engine::{
    Array, DecoderCache, Error, ExpertFusionDecision, ModelTensors, NormWeight, Result, Stream,
    binding::{BoundEmbedding, BoundLinear, adjusted_norm},
    configure_expert_fusion, decode_graph,
    decoder::{LayerContext, LayerLoopOptions, forward_layers},
    fusion_planner::FusionPlanner,
    lowering::{FeedForwardLowering, LayerLowering, MixerLowering},
};

#[derive(Debug)]
pub struct HybridLinearMoeModel {
    pub(super) layers: Vec<HybridLinearMoeLayer>,
    mixers: Vec<MixerLowering>,
    cache_step: usize,
    pub(super) embedding: BoundEmbedding,
    pub(super) output: BoundLinear,
    pub(super) final_norm: NormWeight,
    pub(super) rms_norm_eps: f32,
    pub(super) hidden_size: usize,
    expert_fusion: ExpertFusionDecision,
}

impl HybridLinearMoeModel {
    pub fn load(
        tensors: &ModelTensors,
        decoder: &DecoderConfig,
        bindings: &WeightBindingPlan,
        lowering: &[LayerLowering],
        cache_step: usize,
        stream: &Stream,
    ) -> Result<Self> {
        let compatible = lowering.len() == decoder.num_hidden_layers
            && lowering
                .iter()
                .all(|layer| layer.feed_forward == FeedForwardLowering::SharedRouted)
            && lowering.iter().any(|layer| layer.mixer == MixerLowering::Linear);
        if !compatible || decoder.tie_word_embeddings {
            return Err(Error::InvalidModel(
                "hybrid linear MoE requires untied shared-expert decoder weights".into(),
            ));
        }
        let norm_shift = norm_shift(tensors, bindings, lowering)?;
        let mut layers = Vec::with_capacity(decoder.num_hidden_layers);
        for (index, lowered) in lowering.iter().enumerate() {
            layers.push(HybridLinearMoeLayer::load(
                tensors,
                decoder,
                index,
                bindings.hybrid_decoder_layer(index)?,
                *lowered,
                norm_shift,
                stream,
            )?);
        }
        let boundary = bindings.decoder_boundary()?;
        let expert_fusion = configure_expert_fusion(
            &mut layers,
            stream,
            FusionPlanner::new(stream).expert_mode(FeedForwardLowering::SharedRouted),
        )?;
        Ok(Self {
            layers,
            mixers: lowering.iter().map(|layer| layer.mixer).collect(),
            cache_step,
            embedding: BoundEmbedding::load(tensors, boundary.embedding, stream)?,
            output: BoundLinear::load(tensors, boundary.output, stream)?,
            final_norm: adjusted_norm(tensors, boundary.final_norm, norm_shift, stream)?,
            rms_norm_eps: decoder.rms_norm_eps.to_string().parse()?,
            hidden_size: decoder.hidden_size,
            expert_fusion,
        })
    }

    pub fn new_cache(&self, stream: &Stream) -> Result<DecoderCache> {
        DecoderCache::new_hybrid_linear_with_pool_capacity(
            &self.mixers,
            self.cache_step,
            crate::engine::KvPageFormat::resolve(stream.config().kv_cache.dtype)?,
            stream.config().kv_cache.block_size,
            DecoderCache::physical_page_capacity(stream, self.cache_step),
            stream.paged_arenas(),
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
        hidden: Array,
        cache: &mut DecoderCache,
        position: i32,
        causal: bool,
        positions: Option<&Array>,
        stream: &Stream,
    ) -> Result<Array> {
        let profile = stream.config().diagnostics.profile_layers;
        let profile_graph = stream.config().diagnostics.profile_graph_build;
        let hidden = forward_layers(
            &self.layers,
            hidden,
            cache,
            LayerContext {
                position,
                causal,
                positions,
                image: None,
                stream,
            },
            LayerLoopOptions::new(profile, None, profile_graph),
        )?;
        let output = self.final_norm.apply(&hidden, self.rms_norm_eps, stream)?;
        Ok(output)
    }
}

fn norm_shift(
    tensors: &ModelTensors,
    bindings: &WeightBindingPlan,
    lowering: &[LayerLowering],
) -> Result<f32> {
    let index = lowering
        .iter()
        .position(|layer| layer.mixer == MixerLowering::Linear)
        .ok_or_else(|| Error::InvalidModel("missing linear attention layer".into()))?;
    let layer = bindings.hybrid_decoder_layer(index)?;
    let HybridMixerBindings::Linear(linear) = layer.mixer else {
        return Err(Error::InvalidModel("linear layer has no linear mixer binding".into()));
    };
    let weight = tensors.get(&linear.convolution.source)?;
    let last_dimension = weight.shape()?.last().copied().unwrap_or_default();
    Ok(if last_dimension == 1 {
        0.0
    } else {
        1.0
    })
}
