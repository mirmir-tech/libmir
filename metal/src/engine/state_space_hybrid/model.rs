use architecture::CacheLayout;
use models::{layout::DecoderConfig, semantic::DecoderSpec, weights::WeightBindingPlan};

use super::layer::StateSpaceHybridLayer;
use crate::engine::{
    Array, DecoderCache, Error, ModelTensors, NormWeight, Result, Stream,
    binding::{BoundEmbedding, BoundLinear},
    decode,
    decoder::{LayerContext, LayerLoopOptions, forward_layers, prefill_evaluation_step},
    lowering::LayerLowering,
};

#[derive(Debug)]
pub struct StateSpaceHybridModel {
    pub(super) layers: Vec<StateSpaceHybridLayer>,
    cache_layout: CacheLayout,
    cache_step: usize,
    pub(super) embedding: BoundEmbedding,
    pub(super) output: BoundLinear,
    pub(super) final_norm: NormWeight,
    pub(super) epsilon: f32,
}

impl StateSpaceHybridModel {
    pub fn load(
        tensors: &ModelTensors,
        decoder: &DecoderConfig,
        spec: &DecoderSpec,
        bindings: &WeightBindingPlan,
        lowering: &[LayerLowering],
        cache_step: usize,
        stream: &Stream,
    ) -> Result<Self> {
        if decoder.state_space.is_none() || decoder.tie_word_embeddings {
            return Err(Error::InvalidModel(
                "state-space hybrid runtime requires untied Mamba-2 hybrid decoder weights".into(),
            ));
        }
        let epsilon: f32 = decoder.rms_norm_eps.to_string().parse()?;
        let layers = (0..lowering.len())
            .map(|index| {
                StateSpaceHybridLayer::load(
                    tensors,
                    decoder,
                    spec.layer(index)?,
                    bindings.state_space_hybrid_layer(index)?,
                    epsilon,
                    stream,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let boundary = bindings.decoder_boundary()?;
        Ok(Self {
            layers,
            cache_layout: CacheLayout::for_metal(spec),
            cache_step,
            embedding: BoundEmbedding::load(tensors, boundary.embedding, stream)?,
            output: BoundLinear::load(tensors, boundary.output, stream)?,
            final_norm: NormWeight::load_name(tensors, &boundary.final_norm.source)?,
            epsilon,
        })
    }

    pub fn new_cache(&self, stream: &Stream) -> Result<DecoderCache> {
        DecoderCache::from_layout(
            &self.cache_layout,
            self.cache_step,
            crate::engine::KvPageFormat::resolve(stream.config().kv_cache.dtype)?,
            stream.config().kv_cache.block_size,
            stream.config().kv_cache.block_count as usize,
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
        decode::graph::export_once(&logits, stream)?;
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

    pub(super) fn project_output(&self, hidden: &Array, stream: &Stream) -> Result<Array> {
        self.output.forward(hidden, stream)
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
        self.forward_embedded(hidden, cache, position, causal, stream)
    }

    pub(super) fn forward_embedded(
        &self,
        hidden: Array,
        cache: &mut DecoderCache,
        position: i32,
        causal: bool,
        stream: &Stream,
    ) -> Result<Array> {
        let sequence = usize::try_from(*hidden.shape()?.get(1).ok_or_else(|| {
            Error::InvalidModel("state-space hybrid hidden state has no sequence axis".into())
        })?)?;
        let evaluation_step = causal
            .then(|| {
                Ok::<_, Error>(prefill_evaluation_step(
                    1,
                    sequence,
                    usize::try_from(position)?,
                    self.layers.len(),
                ))
            })
            .transpose()?
            .flatten();
        let hidden = forward_layers(
            &self.layers,
            hidden,
            cache,
            LayerContext {
                position,
                causal,
                positions: None,
                image: None,
                stream,
            },
            LayerLoopOptions::new(
                stream.config().diagnostics.profile_layers,
                evaluation_step,
                stream.config().diagnostics.profile_graph_build,
            ),
        )?;
        self.final_norm.apply(&hidden, self.epsilon, stream)
    }
}
