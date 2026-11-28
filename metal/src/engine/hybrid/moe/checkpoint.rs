//! A real checkpoint's layer, configured through the semantic spec, for the
//! tests that load MLX-LM's legacy Gemma 4 tensor names directly.

use models::{
    layout::DecoderConfig,
    semantic::{DecoderLayerSpec, SemanticModelSpec},
    weights::{TensorCatalog, TensorInfo},
};

use super::{HybridMoeLayer, HybridMoeLayerConfig, HybridMoeModel};
use crate::engine::{ModelTensors, QuantizedEmbedding, Result, Stream, binding::BoundEmbedding};

#[derive(Debug, Clone)]
pub struct CheckpointLayer {
    pub config: HybridMoeLayerConfig,
    pub spec: DecoderLayerSpec,
}

impl CheckpointLayer {
    pub fn discover(decoder: &DecoderConfig, index: usize, group_size: usize) -> Result<Self> {
        let catalog = TensorCatalog {
            tensors: vec![TensorInfo {
                name: "model.layers.0.input_layernorm.weight".into(),
                file: std::path::PathBuf::new(),
                container: models::layout::WeightContainer::SafeTensors,
                dtype: models::weights::DenseDType::Bf16.into(),
                shape: vec![decoder.hidden_size],
                data_start: 0,
                data_offsets: [0, 0],
            }],
        };
        let model = SemanticModelSpec::discover(decoder, &catalog)?;
        let spec = model.decoder.layer(index)?.clone();
        let config = HybridMoeLayerConfig::from_spec(
            &model.decoder,
            &spec,
            decoder.query_key_rows_interleaved,
            group_size,
        )?;
        Ok(Self { config, spec })
    }
}

impl HybridMoeModel {
    /// Every layer of a legacy MLX-LM Gemma 4 checkpoint.
    pub fn load(
        tensors: &ModelTensors,
        decoder: &DecoderConfig,
        group_size: usize,
        cache_step: usize,
        stream: &Stream,
    ) -> Result<Self> {
        let embedding = BoundEmbedding::Affine(QuantizedEmbedding::load(
            tensors,
            "language_model.model.embed_tokens",
            i32::try_from(group_size)?,
        )?);
        let catalog = TensorCatalog {
            tensors: vec![TensorInfo {
                name: "model.layers.0.input_layernorm.weight".into(),
                file: std::path::PathBuf::new(),
                container: models::layout::WeightContainer::SafeTensors,
                dtype: models::weights::DenseDType::Bf16.into(),
                shape: vec![decoder.hidden_size],
                data_start: 0,
                data_offsets: [0, 0],
            }],
        };
        let model = SemanticModelSpec::discover(decoder, &catalog)?;
        let cache_layout = architecture::CacheLayout::for_metal(&model.decoder);
        let mut layers = Vec::with_capacity(decoder.num_hidden_layers);
        for index in 0..decoder.num_hidden_layers {
            let spec = model.decoder.layer(index)?.clone();
            let config = HybridMoeLayerConfig::from_spec(
                &model.decoder,
                &spec,
                decoder.query_key_rows_interleaved,
                group_size,
            )?;
            let layer = CheckpointLayer { config, spec };
            layers.push(HybridMoeLayer::load(tensors, &layer, stream)?);
        }
        Self::finish_load(
            layers,
            cache_layout,
            cache_step,
            embedding,
            tensors.get("language_model.model.norm.weight")?,
            decoder,
            stream,
        )
    }
}
