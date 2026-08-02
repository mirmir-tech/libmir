use models::{layout::PooledVisionConfig, vision::PooledPreprocessedImage, weights::TensorBinding};

use super::{embedding::PatchEmbedding, layer::EncoderLayer, pooler::VisionPooler};
use crate::engine::{Array, Error, ModelTensors, Result, Stream};

#[derive(Debug)]
pub struct PooledVisionTower {
    embedding: PatchEmbedding,
    layers: Vec<EncoderLayer>,
    pooler: VisionPooler,
    patch_width: usize,
    pub(super) bidirectional_image_attention: bool,
}

impl PooledVisionTower {
    pub fn load(
        tensors: &ModelTensors,
        config: &PooledVisionConfig,
        projection: &TensorBinding,
        stream: &Stream,
    ) -> Result<Self> {
        validate_config(config)?;
        let eps = config.rms_norm_eps.to_string().parse::<f32>()?;
        let layers = (0..config.num_hidden_layers)
            .map(|layer| {
                EncoderLayer::load(
                    tensors,
                    &format!("model.vision_tower.encoder.layers.{layer}"),
                    config.num_attention_heads,
                    config.num_key_value_heads,
                    config.head_dim,
                    config.rope_theta,
                    eps,
                    stream,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            embedding: PatchEmbedding::load(tensors, stream)?,
            layers,
            pooler: VisionPooler::load(
                tensors,
                config.hidden_size,
                config.pooling_kernel_size,
                config.standardize,
                eps,
                projection,
                stream,
            )?,
            patch_width: 3 * config.patch_size * config.patch_size,
            bidirectional_image_attention: config.bidirectional_image_attention,
        })
    }

    /// Encodes already extracted RGB patches with explicit `(row, column)`
    /// position IDs.
    ///
    /// Padding must be removed before this call. Every image in the batch must
    /// use the same `grid_height` and `grid_width`; the returned tensor
    /// stays accelerator-resident.
    pub fn forward(
        &self,
        patches: &Array,
        position_ids: &Array,
        grid_height: usize,
        grid_width: usize,
        stream: &Stream,
    ) -> Result<Array> {
        let shape = patches.shape()?;
        if shape.len() != 3 || usize::try_from(shape[2])? != self.patch_width {
            return Err(Error::InvalidModel(format!(
                "pooled vision expected RGB patch width {}, got {shape:?}",
                self.patch_width
            )));
        }
        let mut hidden = self.embedding.forward(patches, position_ids, stream)?;
        for layer in &self.layers {
            hidden = layer.forward(&hidden, position_ids, stream)?;
        }
        self.pooler.forward(&hidden, grid_height, grid_width, stream)
    }

    /// Uploads one preprocessed image and encodes it on the Metal stream.
    pub fn forward_preprocessed(
        &self,
        image: &PooledPreprocessedImage,
        stream: &Stream,
    ) -> Result<Array> {
        let patches =
            image.grid_height.checked_mul(image.grid_width).ok_or(Error::ShapeOverflow)?;
        let patch_values = patches.checked_mul(self.patch_width).ok_or(Error::ShapeOverflow)?;
        if image.patches.len() != patch_values || image.position_ids.len() != patches * 2 {
            return Err(Error::InvalidModel(format!(
                "pooled vision preprocessed image has inconsistent buffers for grid {}x{}",
                image.grid_height, image.grid_width
            )));
        }
        let patches = i32::try_from(patches)?;
        let patch_width = i32::try_from(self.patch_width)?;
        let patch_array = Array::from_f32(&image.patches, &[1, patches, patch_width])?;
        let positions = Array::from_u32(&image.position_ids, &[1, patches, 2])?;
        let output =
            self.forward(&patch_array, &positions, image.grid_height, image.grid_width, stream)?;
        if usize::try_from(output.shape()?[1])? != image.soft_tokens {
            return Err(Error::InvalidModel("pooled vision soft-token count mismatch".into()));
        }
        Ok(output)
    }
}

fn validate_config(config: &PooledVisionConfig) -> Result<()> {
    let projected = config
        .num_attention_heads
        .checked_mul(config.head_dim)
        .ok_or(Error::ShapeOverflow)?;
    if config.hidden_size != projected
        || config.num_key_value_heads == 0
        || !config.num_attention_heads.is_multiple_of(config.num_key_value_heads)
        || config.pooling_kernel_size == 0
        || config.hidden_activation != "gelu_pytorch_tanh"
    {
        return Err(Error::InvalidModel(format!(
            "unsupported pooled vision dimensions or activation: hidden={}, heads={}, kv_heads={}, head_dim={}, pool={}, activation={}",
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            config.pooling_kernel_size,
            config.hidden_activation
        )));
    }
    Ok(())
}
