use models::{
    layout::{EncoderConfig, EncoderRopeScaling, NormKind},
    weights::{TensorCatalog, TensorInfo},
};

use super::CudaSequenceScoringModel;
use crate::{CudaBackend, Error, Result};

impl CudaSequenceScoringModel {
    pub fn load(
        backend: &CudaBackend,
        config: &EncoderConfig,
        catalog: &TensorCatalog,
    ) -> Result<Self> {
        validate(config)?;
        let mut upload = backend.begin_tensor_upload();
        for name in required_names(config) {
            upload.enqueue(required(catalog, &name)?)?;
        }
        Ok(Self {
            backend: backend.clone(),
            config: config.clone(),
            tensors: upload.finish()?,
        })
    }
}

fn validate(config: &EncoderConfig) -> Result<()> {
    if !config.packed_qkv
        || config.norm != NormKind::LayerNorm
        || config.hidden_activation != "gelu"
        || config.num_labels != 1
        || !matches!(config.rope_scaling, Some(EncoderRopeScaling::Ntk { mixed_b: None, .. }))
    {
        return Err(Error::UnsupportedDecoderLayer(
            "CUDA sequence scoring requires packed QKV, LayerNorm, GELU, and fixed NTK RoPE".into(),
        ));
    }
    Ok(())
}

fn required_names(config: &EncoderConfig) -> Vec<String> {
    let mut names: Vec<String> = [
        "new.embeddings.word_embeddings.weight",
        "new.embeddings.token_type_embeddings.weight",
        "new.embeddings.LayerNorm.weight",
        "new.embeddings.LayerNorm.bias",
        "new.pooler.dense.weight",
        "new.pooler.dense.bias",
        "classifier.weight",
        "classifier.bias",
    ]
    .into_iter()
    .map(Into::into)
    .collect();
    for layer in 0..config.num_hidden_layers {
        let prefix = format!("new.encoder.layer.{layer}");
        for suffix in [
            "attention.qkv_proj.weight",
            "attention.qkv_proj.bias",
            "attention.o_proj.weight",
            "attention.o_proj.bias",
            "attn_ln.weight",
            "attn_ln.bias",
            "mlp.up_gate_proj.weight",
            "mlp.down_proj.weight",
            "mlp.down_proj.bias",
            "mlp_ln.weight",
            "mlp_ln.bias",
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
