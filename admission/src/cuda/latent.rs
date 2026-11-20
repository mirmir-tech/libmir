//! Admission of multi-head latent attention decoders (`DeepSeek` V2/V3).

use models::semantic::{
    FeedForwardSpec, MixerSpec, PositionEncodingSpec, RopeScalingSpec, RouterNormalization,
    SemanticModelSpec,
};

use super::{CudaDecoderPlan, CudaDecoderRuntime, CudaFeedForwardLowering, CudaMixerLowering};
use crate::{ArchitectureError, Result};

/// Static shared memory of the latent attention kernel: eight query rows and
/// a 32-row tile of cache rows, each padded by one BF16 pair.
const KERNEL_SHARED_BYTES: usize = 48 * 1024;

/// Every layer is latent attention whose geometry the kernels take, followed
/// by a dense feed-forward or softmax-routed experts with shared experts.
pub(super) fn admit(
    plan: &CudaDecoderPlan,
    semantic: &SemanticModelSpec,
) -> Result<CudaDecoderRuntime> {
    let lowered = plan.layers().iter().all(|layer| {
        layer.mixer == CudaMixerLowering::Latent
            && matches!(
                layer.feed_forward,
                CudaFeedForwardLowering::Dense
                    | CudaFeedForwardLowering::Routed { shared: true, clamped: false }
            )
    });
    if !lowered {
        return Err(ArchitectureError::invalid(
            "CUDA multi-head latent attention requires latent attention in every layer with dense or shared-routed feed-forwards",
        ));
    }
    for layer in &semantic.decoder.layers {
        if let MixerSpec::LatentAttention(latent) = &layer.mixer {
            let row = latent.key_value_rank + latent.rope_head_dim;
            if !latent.key_value_rank.is_multiple_of(64)
                || !latent.rope_head_dim.is_multiple_of(8)
                || (8 + 32) * (row + 2) * 2 > KERNEL_SHARED_BYTES
            {
                return Err(ArchitectureError::invalid(format!(
                    "CUDA multi-head latent attention has no kernel for a {}-wide latent with a {}-wide rotary key",
                    latent.key_value_rank, latent.rope_head_dim
                )));
            }
            let rotary = matches!(
                &latent.position,
                PositionEncodingSpec::Rotary(rotary) if !matches!(
                    rotary.scaling,
                    Some(RopeScalingSpec::PiecewiseFrequency { .. })
                )
            );
            if !rotary {
                return Err(ArchitectureError::invalid(
                    "CUDA multi-head latent attention requires plain, linear or YaRN rotary keys",
                ));
            }
        }
        if let FeedForwardSpec::Routed { routed, .. } = &layer.feed_forward
            && !matches!(
                routed.router_normalization,
                RouterNormalization::SoftmaxTopK | RouterNormalization::SoftmaxProbabilityTopK
            )
        {
            return Err(ArchitectureError::invalid(format!(
                "CUDA multi-head latent attention has no {:?} expert router yet",
                routed.router_normalization
            )));
        }
    }
    Ok(CudaDecoderRuntime::LatentAttention)
}
