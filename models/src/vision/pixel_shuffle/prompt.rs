//! The prompt of one image: its placeholder token replaced by the image's
//! tags around one image token per soft token.

use super::{PixelShufflePreprocessedImage, invalid};
use crate::error::Result;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PixelShufflePromptTokens {
    pub token_ids: Vec<u32>,
    /// The soft tokens' span, between the image tags.
    pub image_start: usize,
    pub image_end: usize,
}

impl PixelShufflePromptTokens {
    /// `input` holds the image token exactly once, where the image goes.
    pub fn prepare(
        input: &[u32],
        image: &PixelShufflePreprocessedImage,
        image_token_id: u32,
        tags: (u32, u32),
    ) -> Result<Self> {
        let mut found = input.iter().enumerate().filter(|(_, token)| **token == image_token_id);
        let (Some((at, _)), None) = (found.next(), found.next()) else {
            return Err(invalid("pixel-shuffle vision prompt needs exactly one image token"));
        };
        let mut token_ids = Vec::with_capacity(input.len() + image.soft_tokens + 1);
        token_ids.extend_from_slice(&input[..at]);
        token_ids.push(tags.0);
        let image_start = token_ids.len();
        token_ids.extend(std::iter::repeat_n(image_token_id, image.soft_tokens));
        let image_end = token_ids.len();
        token_ids.push(tags.1);
        token_ids.extend_from_slice(&input[at + 1..]);
        Ok(Self { token_ids, image_start, image_end })
    }
}
