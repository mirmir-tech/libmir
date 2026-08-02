mod roles;

pub use roles::{EncoderLayerTensorRole, EncoderTensorRole};

use crate::{
    error::{ModelsError, Result},
    layout::{EncoderConfig, EncoderPositionEmbedding},
    weights::{TensorCatalog, TensorStorage},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncoderBindingPlan {
    pub tensors: Vec<EncoderTensorBinding>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncoderTensorBinding {
    pub role: EncoderTensorRole,
    pub source: String,
    pub shape: Vec<usize>,
    pub storage: TensorStorage,
}

impl EncoderTensorBinding {
    #[must_use]
    pub fn physical_sources(&self) -> Vec<&str> {
        let mut sources = vec![self.source.as_str()];
        if let TensorStorage::Dense { bias, .. } = &self.storage {
            sources.extend(bias.as_deref());
        }
        sources
    }
}

impl EncoderBindingPlan {
    pub fn discover(config: &EncoderConfig, catalog: &TensorCatalog) -> Result<Self> {
        let hidden = config.hidden_size;
        let mut tensors = vec![
            binding(
                catalog,
                EncoderTensorRole::WordEmbedding,
                "new.embeddings.word_embeddings",
                vec![config.vocab_size, hidden],
                None,
            )?,
            binding(
                catalog,
                EncoderTensorRole::EmbeddingNorm,
                "new.embeddings.LayerNorm",
                vec![hidden],
                Some(vec![hidden]),
            )?,
        ];
        if config.type_vocab_size > 0 {
            tensors.push(binding(
                catalog,
                EncoderTensorRole::TokenTypeEmbedding,
                "new.embeddings.token_type_embeddings",
                vec![config.type_vocab_size, hidden],
                None,
            )?);
        }
        if config.position_embedding == EncoderPositionEmbedding::Absolute {
            tensors.push(binding(
                catalog,
                EncoderTensorRole::PositionEmbedding,
                "new.embeddings.position_embeddings",
                vec![config.max_position_embeddings, hidden],
                None,
            )?);
        }
        for index in 0..config.num_hidden_layers {
            push_layer(&mut tensors, catalog, config, index)?;
        }
        tensors.extend([
            binding(
                catalog,
                EncoderTensorRole::Pooler,
                "new.pooler.dense",
                vec![hidden, hidden],
                Some(vec![hidden]),
            )?,
            binding(
                catalog,
                EncoderTensorRole::Classifier,
                "classifier",
                vec![config.num_labels, hidden],
                Some(vec![config.num_labels]),
            )?,
        ]);
        Ok(Self { tensors })
    }
}

fn push_layer(
    tensors: &mut Vec<EncoderTensorBinding>,
    catalog: &TensorCatalog,
    config: &EncoderConfig,
    index: usize,
) -> Result<()> {
    let hidden = config.hidden_size;
    let qkv = hidden
        .checked_mul(3)
        .ok_or_else(|| ModelsError::InvalidConfig("encoder QKV width overflow".into()))?;
    let up_gate = config
        .intermediate_size
        .checked_mul(2)
        .ok_or_else(|| ModelsError::InvalidConfig("encoder MLP width overflow".into()))?;
    let prefix = format!("new.encoder.layer.{index}");
    if config.packed_qkv {
        tensors.push(layer(
            catalog,
            index,
            EncoderLayerTensorRole::Qkv,
            &format!("{prefix}.attention.qkv_proj"),
            vec![qkv, hidden],
            Some(vec![qkv]),
        )?);
    } else {
        for (suffix, role) in [
            ("q", EncoderLayerTensorRole::Query),
            ("k", EncoderLayerTensorRole::Key),
            ("v", EncoderLayerTensorRole::Value),
        ] {
            tensors.push(layer(
                catalog,
                index,
                role,
                &format!("{prefix}.attention.{suffix}_proj"),
                vec![hidden, hidden],
                Some(vec![hidden]),
            )?);
        }
    }
    tensors.extend([
        layer(
            catalog,
            index,
            EncoderLayerTensorRole::AttentionOutput,
            &format!("{prefix}.attention.o_proj"),
            vec![hidden, hidden],
            Some(vec![hidden]),
        )?,
        layer(
            catalog,
            index,
            EncoderLayerTensorRole::AttentionNorm,
            &format!("{prefix}.attn_ln"),
            vec![hidden],
            Some(vec![hidden]),
        )?,
        layer(
            catalog,
            index,
            EncoderLayerTensorRole::MlpUpGate,
            &format!("{prefix}.mlp.up_gate_proj"),
            vec![up_gate, hidden],
            None,
        )?,
        layer(
            catalog,
            index,
            EncoderLayerTensorRole::MlpDown,
            &format!("{prefix}.mlp.down_proj"),
            vec![hidden, config.intermediate_size],
            Some(vec![hidden]),
        )?,
        layer(
            catalog,
            index,
            EncoderLayerTensorRole::MlpNorm,
            &format!("{prefix}.mlp_ln"),
            vec![hidden],
            Some(vec![hidden]),
        )?,
    ]);
    Ok(())
}

fn layer(
    catalog: &TensorCatalog,
    index: usize,
    tensor: EncoderLayerTensorRole,
    prefix: &str,
    shape: Vec<usize>,
    bias: Option<Vec<usize>>,
) -> Result<EncoderTensorBinding> {
    binding(catalog, EncoderTensorRole::Layer { index, tensor }, prefix, shape, bias)
}

fn binding(
    catalog: &TensorCatalog,
    role: EncoderTensorRole,
    prefix: &str,
    shape: Vec<usize>,
    bias_shape: Option<Vec<usize>>,
) -> Result<EncoderTensorBinding> {
    let source = format!("{prefix}.weight");
    let tensor = required(catalog, &source, &shape)?;
    let bias = bias_shape
        .map(|shape| {
            let name = format!("{prefix}.bias");
            let bias = required(catalog, &name, &shape)?;
            if bias.dtype != tensor.dtype {
                return Err(ModelsError::InvalidConfig(format!(
                    "encoder tensor `{name}` has dtype {}, expected {}",
                    bias.dtype, tensor.dtype
                )));
            }
            Ok(name)
        })
        .transpose()?;
    Ok(EncoderTensorBinding {
        role,
        source,
        shape,
        storage: TensorStorage::Dense { dtype: tensor.dtype.clone(), bias },
    })
}

fn required<'a>(
    catalog: &'a TensorCatalog,
    name: &str,
    shape: &[usize],
) -> Result<&'a crate::weights::TensorInfo> {
    let tensor = catalog
        .get(name)
        .ok_or_else(|| ModelsError::InvalidConfig(format!("missing encoder tensor `{name}`")))?;
    if tensor.shape != shape {
        return Err(ModelsError::InvalidConfig(format!(
            "encoder tensor `{name}` has shape {:?}, expected {shape:?}",
            tensor.shape
        )));
    }
    Ok(tensor)
}

#[cfg(test)]
mod tests;
