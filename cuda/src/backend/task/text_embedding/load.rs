use models::{
    layout::DecoderConfig,
    weights::{TensorCatalog, TensorInfo, TextTensorLayout},
};

use super::CudaTextEmbeddingModel;
use crate::{CudaBackend, Error, Result};

impl CudaTextEmbeddingModel {
    pub fn load(
        backend: &CudaBackend,
        config: &DecoderConfig,
        catalog: &TensorCatalog,
        layout: &TextTensorLayout,
    ) -> Result<Self> {
        validate(config)?;
        let names = required_names(config, layout);
        let cast = backend.prepare_dense_cast()?;
        let mut upload = backend.begin_tensor_upload();
        for name in &names {
            upload.enqueue_as_bf16(required(catalog, name)?, &cast)?;
        }
        Ok(Self {
            backend: backend.clone(),
            config: config.clone(),
            layout: layout.clone(),
            tensors: upload.finish()?,
        })
    }
}

fn validate(config: &DecoderConfig) -> Result<()> {
    if config.num_hidden_layers == 0
        || config.hidden_size == 0
        || config.intermediate_size == 0
        || config.num_attention_heads == 0
        || config.num_key_value_heads == 0
        || config.head_dim == 0
        || config.hidden_activation.as_deref() != Some("silu")
        || config
            .layer_types
            .iter()
            .any(|layer| *layer != models::layout::AttentionLayerType::Full)
    {
        return Err(Error::UnsupportedDecoderLayer(
            "CUDA embedding requires a dense full-attention SwiGLU backbone".into(),
        ));
    }
    Ok(())
}

fn required_names(config: &DecoderConfig, layout: &TextTensorLayout) -> Vec<String> {
    let mut names = vec![layout.name("embed_tokens.weight"), layout.name("norm.weight")];
    for layer in 0..config.num_hidden_layers {
        let prefix = layout.name(format!("layers.{layer}"));
        for suffix in [
            "input_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "post_attention_layernorm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ] {
            names.push(format!("{prefix}.{suffix}"));
        }
    }
    names
}

fn required<'a>(catalog: &'a TensorCatalog, name: &str) -> Result<&'a TensorInfo> {
    catalog
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .ok_or_else(|| Error::MissingTensor(name.into()))
}
