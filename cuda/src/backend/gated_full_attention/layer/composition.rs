use super::CudaAffineGatedFullAttentionMoeLayer;
use crate::{
    CudaAffineGatedFullAttention, CudaBackend, CudaTensor, Error, Result,
    backend::feed_forward::{FeedForward, LayerNormConfig},
};

impl CudaAffineGatedFullAttentionMoeLayer {
    pub(in crate::backend) fn compose(
        backend: &CudaBackend,
        config: LayerNormConfig,
        attention: CudaAffineGatedFullAttention,
        moe: FeedForward,
        input_norm: CudaTensor,
        post_attention_norm: CudaTensor,
    ) -> Result<Self> {
        if config.hidden_size == 0
            || !config.rms_norm_epsilon.is_finite()
            || config.rms_norm_epsilon < 0.0
            || !config.norm_weight_shift.is_finite()
        {
            return Err(Error::InvalidDecoderKernel("invalid mixed layer normalization"));
        }
        for weight in [&input_norm, &post_attention_norm] {
            if weight.shape() != [config.hidden_size] || weight.as_bf16().is_none() {
                return Err(Error::InvalidDecoderKernel("invalid mixed layer norm weight"));
            }
        }
        Ok(Self {
            backend: backend.clone(),
            config,
            attention,
            moe,
            input_norm,
            post_attention_norm,
        })
    }
}
