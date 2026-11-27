//! The decoders the hybrid runtime executes: linear-attention hybrids with
//! one feed-forward kind, or latent attention decoders whose leading dense
//! layers precede routed ones.

use models::layout::DecoderConfig;

use crate::engine::{
    Error, Result,
    lowering::{Activation, FeedForwardLowering, LayerLowering, MixerLowering},
};

/// Whether the decoder is a latent attention decoder; an error when the
/// runtime cannot execute it.
pub(super) fn check(decoder: &DecoderConfig, lowering: &[LayerLowering]) -> Result<bool> {
    let latent = !lowering.is_empty()
        && lowering.iter().all(|layer| matches!(layer.mixer, MixerLowering::Latent(_)));
    let compatible = lowering.len() == decoder.num_hidden_layers
        && (latent || lowering.iter().all(|layer| layer.feed_forward == lowering[0].feed_forward))
        && lowering.iter().all(|layer| {
            matches!(
                layer.feed_forward,
                FeedForwardLowering::Routed { shared: Some(_), .. }
                    | FeedForwardLowering::Dense {
                        activation: Activation::SwiGlu { clamp: None } | Activation::GeluTanh,
                    }
            )
        })
        && (latent || lowering.iter().any(|layer| layer.mixer == MixerLowering::Linear));
    if compatible {
        Ok(latent)
    } else {
        Err(Error::InvalidModel(
            "hybrid linear runtime requires dense or shared-expert decoder weights".into(),
        ))
    }
}
