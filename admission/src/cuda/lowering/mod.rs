//! The CUDA decoder plan: the shared lowering, refused where CUDA's kernels
//! end, so the limits hold for every runtime that uses those kernels.

use models::semantic::{
    DecoderLayerSpec, MixerSpec, PositionEncodingSpec, QueryTemperature, RopeScalingSpec,
    SemanticModelSpec,
};

use crate::{
    ArchitectureError, Constant, Result, Unlowerable,
    capability::Operations,
    decoder::{DecoderFeatures, Feature, lower_features},
    lowering::{
        Activation, FeedForwardLowering, LatentGeometry, MixerLowering, lower_feed_forward,
    },
};

mod repeated;

pub use repeated::{RepeatedLayer, YarnRotary};

/// Static shared memory of the latent attention kernel: eight query rows and
/// a 32-row tile of cache rows, each padded by one BF16 pair.
const LATENT_SHARED_BYTES: usize = 48 * 1024;
/// The router warp holds 256 experts and ranks one group per lane.
const ROUTER_EXPERTS: usize = 256;
const ROUTER_GROUPS: usize = 32;
/// The clamped `SwiGLU` kernel's sigmoid slope (gpt-oss's).
const CLAMP_ALPHA: f64 = 1.702;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CudaLayer {
    pub index: usize,
    pub mixer: MixerLowering,
    pub feed_forward: FeedForwardLowering,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CudaDecoderPlan {
    layers: Vec<CudaLayer>,
    /// Multi-token prediction modules, loaded only to draft: lowered so that
    /// they too refuse what no kernel computes, but not required of the
    /// runtime.
    prediction: Vec<CudaLayer>,
    features: DecoderFeatures,
    /// Boxed: most plans have none, and the plan travels inside the
    /// admitted architecture.
    repeated: Option<Box<RepeatedLayer>>,
}

impl CudaDecoderPlan {
    /// Lowers every layer; `declared_attention_scale` is the configuration's.
    pub fn lower(spec: &SemanticModelSpec, declared_attention_scale: Option<f64>) -> Result<Self> {
        let decoder = &spec.decoder;
        let layers = decoder.layers.iter().map(lower_layer).collect::<Result<Vec<_>>>()?;
        let repeated = RepeatedLayer::of(spec, &layers).map(Box::new);
        let mut features = lower_features(spec, declared_attention_scale)?;
        if repeated.is_none() {
            features.flags.insert(Feature::NoRepeatedLayer);
        }
        Ok(Self {
            layers,
            prediction: decoder.prediction_layers.iter().map(lower_layer).collect::<Result<_>>()?,
            features,
            repeated,
        })
    }

    /// The layer every layer repeats, which a runtime built around one
    /// layer reads instead of the spec; `None` with
    /// [`Feature::NoRepeatedLayer`].
    #[must_use]
    pub fn repeated(&self) -> Option<&RepeatedLayer> {
        self.repeated.as_deref()
    }

    #[must_use]
    pub fn prediction(&self) -> &[CudaLayer] {
        &self.prediction
    }

    #[must_use]
    pub fn layers(&self) -> &[CudaLayer] {
        &self.layers
    }

    #[must_use]
    pub const fn features(&self) -> DecoderFeatures {
        self.features
    }

    /// What a runtime must execute to run this plan.
    #[must_use]
    pub fn requirements(&self) -> Operations {
        Operations::required(
            self.layers.iter().map(|layer| (&layer.mixer, &layer.feed_forward)),
            self.features,
        )
    }
}

fn lower_layer(layer: &DecoderLayerSpec) -> Result<CudaLayer> {
    let refuse = |feature| ArchitectureError::Unlowerable { layer: Some(layer.index), feature };
    let mixer = MixerLowering::lower(&layer.mixer);
    if let MixerLowering::Latent(geometry) = mixer {
        latent_kernel(geometry).map_err(refuse)?;
    }
    if let MixerSpec::LatentAttention(latent) = &layer.mixer {
        // The latent kernels rotate a key part, and not by Llama 3's blend.
        let rotates = matches!(&latent.position, PositionEncodingSpec::Rotary(rotary)
            if !matches!(rotary.scaling, Some(RopeScalingSpec::PiecewiseFrequency { .. })));
        if !rotates {
            return Err(refuse(Unlowerable::LatentRotary));
        }
        latent_query_temperature(latent.query_temperature).map_err(refuse)?;
    }
    let feed_forward = lower_feed_forward(&layer.feed_forward, layer.index)?;
    router_kernel(&feed_forward).map_err(refuse)?;
    Ok(CudaLayer { index: layer.index, mixer, feed_forward })
}

/// The latent kernels count tempered positions from zero (Mistral 4), not
/// from one (Llama 4); the CUDA latent loader checks the same.
pub fn latent_query_temperature(
    temperature: Option<QueryTemperature>,
) -> std::result::Result<(), Unlowerable> {
    match temperature {
        Some(QueryTemperature { offset, .. }) if offset != 0 => {
            Err(Unlowerable::LatentQueryTemperatureOffset { offset })
        },
        _ => Ok(()),
    }
}

fn latent_kernel(geometry: LatentGeometry) -> std::result::Result<(), Unlowerable> {
    let LatentGeometry { key_value_rank, rope_head_dim } = geometry;
    let row = key_value_rank + rope_head_dim;
    if key_value_rank.is_multiple_of(64)
        && rope_head_dim.is_multiple_of(8)
        && (8 + 32) * (row + 2) * 2 <= LATENT_SHARED_BYTES
    {
        Ok(())
    } else {
        Err(Unlowerable::LatentGeometry { key_value_rank, rope_head_dim })
    }
}

fn router_kernel(feed_forward: &FeedForwardLowering) -> std::result::Result<(), Unlowerable> {
    if let FeedForwardLowering::LatentRouted { latent_size: None, .. } = feed_forward {
        return Err(Unlowerable::LatentExpertsWithoutLatentSize);
    }
    let Some(experts) = feed_forward.experts() else {
        return Ok(());
    };
    if let Activation::SwiGlu { clamp: Some(clamp) } = experts.activation
        && (clamp.alpha - CLAMP_ALPHA).abs() > 1e-6
    {
        return Err(Unlowerable::ClampAlpha { alpha: Constant(clamp.alpha) });
    }
    if experts.count > ROUTER_EXPERTS {
        return Err(Unlowerable::ExpertCount {
            experts: experts.count,
            max: ROUTER_EXPERTS,
        });
    }
    match experts.group_limit {
        Some(limit) if limit.groups > ROUTER_GROUPS => {
            Err(Unlowerable::ExpertGroups { groups: limit.groups, max: ROUTER_GROUPS })
        },
        _ => Ok(()),
    }
}
