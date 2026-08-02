use models::{
    layout::{EncoderConfig, EncoderPositionEmbedding, EncoderRopeScaling, NormKind},
    weights::{EncoderBindingPlan, TensorCatalog, TensorInfo},
};

use super::CudaSequenceScoringModel;
use crate::{CudaBackend, Error, Result};

impl CudaSequenceScoringModel {
    pub fn load(
        backend: &CudaBackend,
        config: &EncoderConfig,
        catalog: &TensorCatalog,
        bindings: &EncoderBindingPlan,
    ) -> Result<Self> {
        validate(config)?;
        let mut upload = backend.begin_tensor_upload();
        for name in bindings
            .tensors
            .iter()
            .flat_map(models::weights::EncoderTensorBinding::physical_sources)
        {
            upload.enqueue(required(catalog, name)?)?;
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
        || config.position_embedding != EncoderPositionEmbedding::Rope
        || config.num_labels != 1
        || config.type_vocab_size == 0
        || !matches!(config.rope_scaling, Some(EncoderRopeScaling::Ntk { mixed_b: None, .. }))
    {
        return Err(Error::UnsupportedDecoderLayer(
            "CUDA sequence scoring requires packed QKV, LayerNorm, GELU, and fixed NTK RoPE".into(),
        ));
    }
    Ok(())
}

fn required<'a>(catalog: &'a TensorCatalog, name: &str) -> Result<&'a TensorInfo> {
    catalog
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .ok_or_else(|| Error::MissingTensor(name.into()))
}
