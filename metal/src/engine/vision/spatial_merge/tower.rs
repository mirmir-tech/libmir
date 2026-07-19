use models::{layout::SpatialMergeVisionConfig, vision::SpatialMergePreprocessedImage};

use super::{dimension, embedding::PatchEmbedding, layer::EncoderLayer, merger::PatchMerger};
use crate::engine::{Array, Error, ModelTensors, Result, Stream};

#[derive(Debug)]
pub struct SpatialMergeVisionTower {
    embedding: PatchEmbedding,
    layers: Vec<EncoderLayer>,
    merger: PatchMerger,
    patch_width: usize,
    output_hidden_size: usize,
    merge: usize,
}

impl SpatialMergeVisionTower {
    pub fn load(
        tensors: &ModelTensors,
        config: &SpatialMergeVisionConfig,
        stream: &Stream,
    ) -> Result<Self> {
        validate_config(config)?;
        let prefix = vision_prefix(tensors)?;
        let head_dim = config.hidden_size / config.num_attention_heads;
        let layers = (0..config.num_hidden_layers)
            .map(|layer| {
                EncoderLayer::load(
                    tensors,
                    &format!("{prefix}.blocks.{layer}"),
                    config.num_attention_heads,
                    head_dim,
                    stream,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            embedding: PatchEmbedding::load(
                tensors,
                config.hidden_size,
                config.num_position_embeddings,
                config.spatial_merge_size,
                prefix,
                stream,
            )?,
            layers,
            merger: PatchMerger::load(
                tensors,
                config.hidden_size,
                config.spatial_merge_size,
                prefix,
                stream,
            )?,
            patch_width: config.in_channels
                * config.temporal_patch_size
                * config.patch_size
                * config.patch_size,
            output_hidden_size: config.output_hidden_size,
            merge: config.spatial_merge_size,
        })
    }

    pub fn forward_preprocessed(
        &self,
        image: &SpatialMergePreprocessedImage,
        stream: &Stream,
    ) -> Result<Array> {
        if image.grid_t != 1 {
            return Err(Error::InvalidModel(
                "spatial-merge vision Metal MVP accepts one still image, not video".into(),
            ));
        }
        let sequence =
            image.grid_height.checked_mul(image.grid_width).ok_or(Error::ShapeOverflow)?;
        if image.patches.len() != sequence * self.patch_width {
            return Err(Error::InvalidModel(
                "spatial-merge vision preprocessed patch buffer has an invalid length".into(),
            ));
        }
        let patches = Array::from_f32(
            &image.patches,
            &[
                1,
                dimension(sequence, "patch sequence")?,
                dimension(self.patch_width, "patch width")?,
            ],
        )?;
        let positions = position_ids(image, self.merge)?;
        let mut hidden = self.embedding.forward(&patches, image, stream)?;
        for layer in &self.layers {
            hidden = layer.forward(&hidden, &positions, stream)?;
        }
        let output = self.merger.forward(&hidden, stream)?;
        let expected = [
            1,
            dimension(image.soft_tokens, "soft tokens")?,
            dimension(self.output_hidden_size, "output hidden size")?,
        ];
        if output.shape()? != expected {
            return Err(Error::InvalidModel(format!(
                "spatial-merge vision tower output {:?} does not match {expected:?}",
                output.shape()?
            )));
        }
        Ok(output)
    }
}

fn vision_prefix(tensors: &ModelTensors) -> Result<&'static str> {
    for prefix in ["model.visual", "vision_tower"] {
        if tensors.contains(&format!("{prefix}.patch_embed.proj.weight"))? {
            return Ok(prefix);
        }
    }
    Err(Error::MissingTensor(
        "model.visual.patch_embed.proj.weight or vision_tower.patch_embed.proj.weight".into(),
    ))
}

fn position_ids(image: &SpatialMergePreprocessedImage, merge: usize) -> Result<Array> {
    let mut values = Vec::with_capacity(image.grid_height * image.grid_width * 2);
    for block_y in 0..image.grid_height / merge {
        for block_x in 0..image.grid_width / merge {
            for merge_y in 0..merge {
                for merge_x in 0..merge {
                    values.push(u32::try_from(block_y * merge + merge_y)?);
                    values.push(u32::try_from(block_x * merge + merge_x)?);
                }
            }
        }
    }
    Array::from_u32(
        &values,
        &[dimension(image.grid_height * image.grid_width, "position count")?, 2],
    )
}

fn validate_config(config: &SpatialMergeVisionConfig) -> Result<()> {
    if config.hidden_size == 0
        || config.num_attention_heads == 0
        || !config.hidden_size.is_multiple_of(config.num_attention_heads)
        || config.in_channels != 3
        || config.spatial_merge_size == 0
        || config.hidden_activation != "gelu_pytorch_tanh"
    {
        return Err(Error::InvalidModel(format!(
            "unsupported spatial-merge vision configuration: {config:?}"
        )));
    }
    Ok(())
}
