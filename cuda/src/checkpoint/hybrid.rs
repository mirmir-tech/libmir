use std::collections::HashSet;

use models::{
    layout::{AttentionLayerType, DecoderConfig},
    weights::{TensorCatalog, TensorInfo},
};

use super::{HybridLinearModelLoadConfig, model::payload_bytes};
use crate::{CudaBackend, CudaHybridLinearModelTemplate, Error, Result};

impl CudaBackend {
    pub fn load_hybrid_linear_model_template(
        &self,
        decoder: &DecoderConfig,
        catalog: &TensorCatalog,
        load: HybridLinearModelLoadConfig,
    ) -> Result<CudaHybridLinearModelTemplate> {
        let mut ignored = |_completed, _detail| {};
        self.load_hybrid_linear_model_template_with_progress(decoder, catalog, load, &mut ignored)
    }

    pub(crate) fn load_hybrid_linear_model_template_with_progress(
        &self,
        decoder: &DecoderConfig,
        catalog: &TensorCatalog,
        load: HybridLinearModelLoadConfig,
        progress: &mut dyn FnMut(u64, String),
    ) -> Result<CudaHybridLinearModelTemplate> {
        if load.max_sequence_blocks == 0 {
            return Err(Error::UnsupportedDecoderLayer(
                "hybrid model sequence block capacity is empty".into(),
            ));
        }
        let source = HybridSource::discover(decoder, catalog)?;
        let mut upload = self.begin_tensor_upload();
        for tensor in &source {
            upload.enqueue(tensor)?;
        }
        let tensors = upload.finish()?;
        let bytes = payload_bytes(source.iter().copied())?;
        progress(bytes, format!("uploaded {} affine hybrid tensors", source.len()));
        tracing::debug!(
            backend = "cuda",
            layers = decoder.num_hidden_layers,
            tensors = source.len(),
            bytes,
            "loaded affine hybrid linear/full-attention model template"
        );
        CudaHybridLinearModelTemplate::from_tensors(
            self,
            decoder,
            &tensors,
            load.cache,
            load.max_sequence_blocks,
        )
    }
}

struct HybridSource {
    names: Vec<String>,
}

impl HybridSource {
    fn discover<'a>(
        decoder: &DecoderConfig,
        catalog: &'a TensorCatalog,
    ) -> Result<Vec<&'a TensorInfo>> {
        let mut source = Self { names: Vec::new() };
        source.affine("language_model.model.embed_tokens");
        source.raw("language_model.model.norm.weight");
        if !decoder.tie_word_embeddings {
            source.affine("language_model.lm_head");
        }
        for layer in 0..decoder.num_hidden_layers {
            source.layer(decoder, layer)?;
        }
        let mut seen = HashSet::new();
        source
            .names
            .iter()
            .filter(|name| seen.insert((*name).clone()))
            .map(|name| required(catalog, name))
            .collect()
    }

    fn layer(&mut self, decoder: &DecoderConfig, layer: usize) -> Result<()> {
        let prefix = format!("language_model.model.layers.{layer}");
        self.raw(&format!("{prefix}.input_layernorm.weight"));
        self.raw(&format!("{prefix}.post_attention_layernorm.weight"));
        self.moe(&format!("{prefix}.mlp"));
        match decoder.layer_type(layer) {
            AttentionLayerType::Linear => self.linear(&format!("{prefix}.linear_attn")),
            AttentionLayerType::Full => self.full(&format!("{prefix}.self_attn")),
            AttentionLayerType::Sliding => {
                return Err(Error::UnsupportedDecoderLayer(
                    "hybrid checkpoint contains sliding attention".into(),
                ));
            },
        }
        Ok(())
    }

    fn moe(&mut self, prefix: &str) {
        for name in [
            "gate",
            "switch_mlp.gate_proj",
            "switch_mlp.up_proj",
            "switch_mlp.down_proj",
            "shared_expert.gate_proj",
            "shared_expert.up_proj",
            "shared_expert.down_proj",
            "shared_expert_gate",
        ] {
            self.affine(&format!("{prefix}.{name}"));
        }
    }

    fn linear(&mut self, prefix: &str) {
        for name in ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"] {
            self.affine(&format!("{prefix}.{name}"));
        }
        for name in ["conv1d.weight", "norm.weight", "A_log", "dt_bias"] {
            self.raw(&format!("{prefix}.{name}"));
        }
    }

    fn full(&mut self, prefix: &str) {
        for name in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            self.affine(&format!("{prefix}.{name}"));
        }
        for name in ["q_norm.weight", "k_norm.weight"] {
            self.raw(&format!("{prefix}.{name}"));
        }
    }

    fn affine(&mut self, prefix: &str) {
        for suffix in ["weight", "scales", "biases"] {
            self.raw(&format!("{prefix}.{suffix}"));
        }
    }

    fn raw(&mut self, name: &str) {
        self.names.push(name.into());
    }
}

fn required<'a>(catalog: &'a TensorCatalog, name: &str) -> Result<&'a TensorInfo> {
    catalog
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .ok_or_else(|| Error::MissingTensor(name.into()))
}
