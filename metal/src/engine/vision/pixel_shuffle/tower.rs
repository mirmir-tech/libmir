use models::{
    layout::{HiddenActivation, PixelShuffleVisionConfig},
    vision::PixelShufflePreprocessedImage,
};

use super::{block::EncoderBlock, dimension, embedding::PatchEmbedding, projector::Projector};
use crate::engine::{Array, Error, ModelTensors, Result, Stream};

#[derive(Debug)]
pub struct PixelShuffleVisionTower {
    embedding: PatchEmbedding,
    blocks: Vec<EncoderBlock>,
    projector: Projector,
    patch_width: usize,
    factor: usize,
    output_hidden_size: usize,
}

impl PixelShuffleVisionTower {
    pub fn load(
        tensors: &ModelTensors,
        config: &PixelShuffleVisionConfig,
        stream: &Stream,
    ) -> Result<Self> {
        validate_config(config)?;
        let blocks = (0..config.num_hidden_layers)
            .map(|index| EncoderBlock::load(tensors, index, config, stream))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            embedding: PatchEmbedding::load(tensors, config, stream)?,
            blocks,
            projector: Projector::load(tensors, config, stream)?,
            patch_width: config.num_channels * config.patch_size * config.patch_size,
            factor: config.shuffle_factor,
            output_hidden_size: config.output_hidden_size,
        })
    }

    /// Soft tokens `[1, soft_tokens, output_hidden_size]` for one image.
    pub fn forward_preprocessed(
        &self,
        image: &PixelShufflePreprocessedImage,
        stream: &Stream,
    ) -> Result<Array> {
        let (rows, cols) = (image.grid_height, image.grid_width);
        let patches = rows.checked_mul(cols).ok_or(Error::ShapeOverflow)?;
        if rows == 0
            || !rows.is_multiple_of(self.factor)
            || !cols.is_multiple_of(self.factor)
            || image.patches.len() != patches * self.patch_width
            || image.soft_tokens != (rows / self.factor) * (cols / self.factor)
        {
            return Err(Error::InvalidModel(
                "inconsistent pixel-shuffle preprocessed image".into(),
            ));
        }
        let input = Array::from_f32(
            &image.patches,
            &[1, dimension(patches, "patch count")?, dimension(self.patch_width, "patch width")?],
        )?;
        let mut hidden = self.embedding.forward(&input, rows, cols, stream)?;
        for block in &self.blocks {
            hidden = block.forward(&hidden, stream)?;
        }
        let output = self.projector.forward(&hidden, rows, cols, stream)?;
        let expected = [
            1,
            dimension(image.soft_tokens, "soft tokens")?,
            dimension(self.output_hidden_size, "output hidden size")?,
        ];
        if output.shape()? != expected {
            return Err(Error::InvalidModel(format!(
                "pixel-shuffle vision tower output {:?} does not match {expected:?}",
                output.shape()?
            )));
        }
        Ok(output)
    }
}

fn validate_config(config: &PixelShuffleVisionConfig) -> Result<()> {
    let heads = config.num_attention_heads;
    if config.hidden_size == 0
        || heads == 0
        || !config.hidden_size.is_multiple_of(heads)
        || config.num_channels != 3
        || config.shuffle_factor == 0
        || config.position_grid == 0
        || config.hidden_activation != HiddenActivation::Gelu
    {
        return Err(Error::InvalidModel(format!(
            "unsupported pixel-shuffle vision geometry or activation: hidden={}, heads={heads}, factor={}, activation={}",
            config.hidden_size, config.shuffle_factor, config.hidden_activation
        )));
    }
    Ok(())
}
