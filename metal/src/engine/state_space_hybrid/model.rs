use models::{layout::DecoderConfig, weights::WeightBindingPlan};

use super::layer::StateSpaceHybridLayer;
use crate::engine::{
    Array, DecoderCache, Error, ModelTensors, NormWeight, Result, Stream,
    binding::{BoundEmbedding, BoundLinear},
    decode_graph,
    decoder::{LayerContext, LayerLoopOptions, forward_layers, prefill_evaluation_step},
    lowering::{LayerLowering, MixerLowering},
};

#[derive(Debug)]
pub struct StateSpaceHybridModel {
    pub(super) layers: Vec<StateSpaceHybridLayer>,
    mixers: Vec<MixerLowering>,
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
                    index,
                    bindings.state_space_hybrid_layer(index)?,
                    epsilon,
                    stream,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let boundary = bindings.decoder_boundary()?;
        Ok(Self {
            layers,
            mixers: lowering.iter().map(|layer| layer.mixer).collect(),
            cache_step,
            embedding: BoundEmbedding::load(tensors, boundary.embedding, stream)?,
            output: BoundLinear::load(tensors, boundary.output, stream)?,
            final_norm: NormWeight::load_name(tensors, &boundary.final_norm.source)?,
            epsilon,
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
