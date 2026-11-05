//! The compatibility runners still implement some operations directly. Reject
//! contracts they cannot execute instead of silently ignoring sidecar
//! overrides.
use models::{
    layout::DecoderConfig,
    semantic::{MixerSpec, QkNormalization, SemanticModelSpec},
};

use super::MetalDecoderRuntime;
use crate::{ArchitectureError, Result};

#[allow(
    clippy::float_cmp,
    reason = "compatibility kernels must execute the declared constants exactly"
)]
pub(super) fn validate(
    decoder: &DecoderConfig,
    spec: &SemanticModelSpec,
    runtime: MetalDecoderRuntime,
) -> Result<()> {
    let embedding = if runtime == MetalDecoderRuntime::DenseAndRouted {
        f64::from(
            u32::try_from(decoder.hidden_size)
                .map_err(|_| ArchitectureError::invalid("Metal hidden size exceeds u32"))?,
        )
        .sqrt()
    } else {
        1.0
    };
    if spec.decoder.embedding_multiplier != embedding {
        return Err(ArchitectureError::invalid(format!(
            "Metal {runtime:?} requires embedding_multiplier={embedding}; requested {}",
            spec.decoder.embedding_multiplier
        )));
    }
    for layer in &spec.decoder.layers {
        let MixerSpec::SoftmaxAttention(attention) = &layer.mixer else {
            continue;
        };
        let normalization_supported = match runtime {
            MetalDecoderRuntime::Dense => matches!(
                attention.qk_normalization,
                QkNormalization::None | QkNormalization::QueryKeyRms
            ),
            MetalDecoderRuntime::DenseAndRouted => {
                attention.qk_normalization == QkNormalization::QueryKeyValueRms
            },
            MetalDecoderRuntime::SharedRouted => {
                attention.qk_normalization == QkNormalization::QueryKeyRms
            },
            MetalDecoderRuntime::ClampedRouted => {
                attention.qk_normalization == QkNormalization::None
            },
        };
        if !normalization_supported {
            return Err(ArchitectureError::invalid(format!(
                "Metal {runtime:?} cannot execute layer {} normalization {:?}",
                layer.index, attention.qk_normalization
            )));
        }
        let head_dim = u32::try_from(decoder.layer_head_dim(layer.index))
            .map_err(|_| ArchitectureError::invalid("Metal head dimension exceeds u32"))?;
        let scale = match runtime {
            MetalDecoderRuntime::DenseAndRouted => 1.0,
            MetalDecoderRuntime::ClampedRouted => {
                decoder.attention_scale.unwrap_or_else(|| f64::from(head_dim).sqrt().recip())
            },
            MetalDecoderRuntime::Dense | MetalDecoderRuntime::SharedRouted => {
                f64::from(head_dim).sqrt().recip()
            },
        };
        if attention.scale != scale {
            return Err(ArchitectureError::invalid(format!(
                "Metal {runtime:?} requires layer {} attention scale={scale}; requested {}",
                layer.index, attention.scale
            )));
        }
    }
    Ok(())
}
