use super::PooledPreprocessedImage;
use crate::{
    error::{ModelsError, Result},
    layout::PooledVisionConfig,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PooledPromptTokens {
    pub token_ids: Vec<u32>,
    pub image_start: usize,
    pub image_end: usize,
}

impl PooledPromptTokens {
    pub fn prepare(
        input: &[u32],
        image: &PooledPreprocessedImage,
        config: &PooledVisionConfig,
    ) -> Result<Self> {
        if image.soft_tokens == 0 {
            return Err(invalid("pooled vision image produced no soft tokens"));
        }
        let mut placeholders = input
            .iter()
            .enumerate()
            .filter(|(_, token)| **token == config.image_token_id)
            .map(|(index, _)| index);
        let placeholder = placeholders
            .next()
            .ok_or_else(|| invalid("pooled vision prompt has no image placeholder"))?;
        if placeholders.next().is_some() {
            return Err(invalid("pooled vision MVP requires exactly one image placeholder"));
        }
        let added = image
            .soft_tokens
            .checked_add(1)
            .ok_or_else(|| invalid("pooled vision expanded image placeholder length overflowed"))?;
        let mut token_ids = Vec::with_capacity(input.len().saturating_add(added));
        token_ids.extend_from_slice(&input[..placeholder]);
        token_ids.push(config.image_begin_token_id);
        let image_start = token_ids.len();
        token_ids.extend(std::iter::repeat_n(config.image_token_id, image.soft_tokens));
        let image_end = token_ids.len();
        token_ids.push(config.image_end_token_id);
        token_ids.extend_from_slice(&input[placeholder + 1..]);
        Ok(Self { token_ids, image_start, image_end })
    }
}

fn invalid(message: impl Into<String>) -> ModelsError {
    ModelsError::InvalidConfig(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expands_one_placeholder_and_tracks_only_soft_tokens() -> Result<()> {
        let prompt = PooledPromptTokens::prepare(&[7, 10, 8], &image(3), &config())?;
        assert_eq!(prompt.token_ids, [7, 11, 10, 10, 10, 12, 8]);
        assert_eq!((prompt.image_start, prompt.image_end), (2, 5));
        Ok(())
    }

    #[test]
    fn rejects_ambiguous_image_layout() {
        assert!(PooledPromptTokens::prepare(&[10, 10], &image(3), &config()).is_err());
    }

    fn image(soft_tokens: usize) -> PooledPreprocessedImage {
        PooledPreprocessedImage {
            patches: Vec::new(),
            position_ids: Vec::new(),
            grid_height: 0,
            grid_width: 0,
            soft_tokens,
        }
    }

    fn config() -> PooledVisionConfig {
        PooledVisionConfig {
            hidden_size: 4,
            output_hidden_size: 4,
            intermediate_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 4,
            patch_size: 2,
            pooling_kernel_size: 1,
            position_embedding_size: 2,
            rms_norm_eps: 1.0e-6,
            rope_theta: 10_000.0,
            hidden_activation: "gelu_pytorch_tanh".into(),
            image_token_id: 10,
            image_begin_token_id: 11,
            image_end_token_id: 12,
            soft_tokens_per_image: 3,
            standardize: false,
            bidirectional_image_attention: true,
            use_clipped_linears: false,
        }
    }
}
