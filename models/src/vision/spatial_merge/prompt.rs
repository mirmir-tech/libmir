use super::SpatialMergePreprocessedImage;
use crate::{
    error::{ModelsError, Result},
    layout::SpatialMergeVisionConfig,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpatialMergePromptTokens {
    pub token_ids: Vec<u32>,
    pub image_start: usize,
    pub image_end: usize,
    pub position_ids: Vec<u32>,
    pub position_delta: i32,
}

impl SpatialMergePromptTokens {
    pub fn prepare(
        input: &[u32],
        image: &SpatialMergePreprocessedImage,
        config: &SpatialMergeVisionConfig,
    ) -> Result<Self> {
        let placeholders = input
            .iter()
            .enumerate()
            .filter(|(_, token)| **token == config.image_token_id)
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        if placeholders.len() != 1 || image.soft_tokens == 0 {
            return Err(invalid(
                "spatial-merge vision MVP requires exactly one placeholder and a nonempty image",
            ));
        }
        let placeholder = placeholders[0];
        let mut token_ids = Vec::with_capacity(input.len() + image.soft_tokens + 1);
        token_ids.extend_from_slice(&input[..placeholder]);
        token_ids.push(config.vision_start_token_id);
        let image_start = token_ids.len();
        token_ids.extend(std::iter::repeat_n(config.image_token_id, image.soft_tokens));
        let image_end = token_ids.len();
        token_ids.push(config.vision_end_token_id);
        token_ids.extend_from_slice(&input[placeholder + 1..]);
        let (position_ids, position_delta) = positions(
            token_ids.len(),
            image_start,
            image_end,
            image.grid_height / config.spatial_merge_size,
            image.grid_width / config.spatial_merge_size,
        )?;
        Ok(Self {
            token_ids,
            image_start,
            image_end,
            position_ids,
            position_delta,
        })
    }
}

fn positions(
    sequence: usize,
    image_start: usize,
    image_end: usize,
    grid_height: usize,
    grid_width: usize,
) -> Result<(Vec<u32>, i32)> {
    if grid_height * grid_width != image_end - image_start {
        return Err(invalid("spatial-merge vision image grid does not match its soft-token span"));
    }
    let mut axes = (0..3).map(|_| Vec::with_capacity(sequence)).collect::<Vec<_>>();
    for position in 0..image_start {
        let position = u32::try_from(position)?;
        for axis in &mut axes {
            axis.push(position);
        }
    }
    let base = u32::try_from(image_start)?;
    for y in 0..grid_height {
        for x in 0..grid_width {
            axes[0].push(base);
            axes[1].push(base + u32::try_from(y)?);
            axes[2].push(base + u32::try_from(x)?);
        }
    }
    let image_max = base + u32::try_from(grid_height.max(grid_width).saturating_sub(1))?;
    let suffix_start = image_max + 1;
    for offset in 0..sequence - image_end {
        let position = suffix_start + u32::try_from(offset)?;
        for axis in &mut axes {
            axis.push(position);
        }
    }
    let maximum = axes.iter().flatten().copied().max().unwrap_or(0);
    let delta = i64::from(maximum) + 1 - i64::try_from(sequence)?;
    Ok((axes.into_iter().flatten().collect(), i32::try_from(delta)?))
}

fn invalid(message: impl Into<String>) -> ModelsError {
    ModelsError::InvalidConfig(message.into())
}
