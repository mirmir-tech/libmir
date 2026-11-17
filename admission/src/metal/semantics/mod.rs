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
    // The dense runtime applies any declared embedding factor (Gemma 3's
    // sqrt(hidden)); it does not add norm weight offsets, which a GGUF import
    // has already folded into the stored weights.
    if runtime == MetalDecoderRuntime::Dense {
        let offset = std::iter::once(spec.decoder.final_norm.weight_offset).chain(
            spec.decoder.layers.iter().flat_map(|layer| {
                [layer.input_norm.weight_offset, layer.post_attention_norm.weight_offset]
            }),
        );
        if offset.into_iter().any(|offset| offset != 0.0) {
            return Err(ArchitectureError::invalid(
                "Metal Dense cannot execute norm weight offsets",
            ));
        }
    }
    let embedding = if runtime == MetalDecoderRuntime::DenseAndRouted {
        f64::from(
            u32::try_from(decoder.hidden_size)
                .map_err(|_| ArchitectureError::invalid("Metal hidden size exceeds u32"))?,
        )
        .sqrt()
    } else {
        1.0
    };
    if runtime != MetalDecoderRuntime::Dense && spec.decoder.embedding_multiplier != embedding {
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
            MetalDecoderRuntime::ClampedRouted | MetalDecoderRuntime::Dense => {
                decoder.attention_scale.unwrap_or_else(|| f64::from(head_dim).sqrt().recip())
            },
            MetalDecoderRuntime::SharedRouted => f64::from(head_dim).sqrt().recip(),
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
