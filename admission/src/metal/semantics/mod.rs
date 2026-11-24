//! The compatibility runners still implement some operations directly. Reject
//! contracts they cannot execute instead of silently ignoring sidecar
//! overrides.
use models::{
    layout::DecoderConfig,
    semantic::{
        AttentionWindow, MixerSpec, NormalizationKind, PositionEncodingSpec, QkNormalization,
        RopeScalingSpec, SemanticModelSpec,
    },
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
    if runtime != MetalDecoderRuntime::DenseGelu && crate::uses_layer_sharing(spec) {
        return Err(ArchitectureError::invalid(format!(
            "Metal {runtime:?} does not run per-layer inputs or layers sharing keys and values"
        )));
    }
    // Only the dense runtime adds Gemma's unit norm offset (the GGUF import
    // has already folded it into the stored weights); it also applies any
    // declared embedding factor (Gemma 3's sqrt(hidden)).
    if runtime != MetalDecoderRuntime::Dense {
        let offset = std::iter::once(spec.decoder.final_norm.weight_offset).chain(
            spec.decoder.layers.iter().flat_map(|layer| {
                [layer.input_norm.weight_offset, layer.post_attention_norm.weight_offset]
            }),
        );
        if offset.into_iter().any(|offset| offset != 0.0) {
            return Err(ArchitectureError::invalid(format!(
                "Metal {runtime:?} cannot execute norm weight offsets"
            )));
        }
    }
    let embedding = if matches!(
        runtime,
        MetalDecoderRuntime::DenseAndRouted | MetalDecoderRuntime::DenseGelu
    ) {
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
    if runtime != MetalDecoderRuntime::Dense {
        dense_only(spec, runtime)?;
    }
    for layer in &spec.decoder.layers {
        let MixerSpec::SoftmaxAttention(attention) = &layer.mixer else {
            continue;
        };
        let normalization_supported = match runtime {
            MetalDecoderRuntime::Dense => matches!(
                attention.qk_normalization,
                QkNormalization::None
                    | QkNormalization::QueryKeyRms
                    | QkNormalization::QueryKeyRmsAcrossHeads
                    | QkNormalization::QueryKeyUnitRms
            ),
            MetalDecoderRuntime::DenseAndRouted | MetalDecoderRuntime::DenseGelu => {
                attention.qk_normalization == QkNormalization::QueryKeyValueRms
            },
            MetalDecoderRuntime::SharedRouted | MetalDecoderRuntime::DenseMixed => {
                attention.qk_normalization == QkNormalization::QueryKeyRms
            },
            MetalDecoderRuntime::ClampedRouted
            | MetalDecoderRuntime::StateSpaceHybrid
            | MetalDecoderRuntime::LatentAttention => {
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
            MetalDecoderRuntime::DenseAndRouted | MetalDecoderRuntime::DenseGelu => 1.0,
            MetalDecoderRuntime::ClampedRouted
            | MetalDecoderRuntime::StateSpaceHybrid
            | MetalDecoderRuntime::LatentAttention
            | MetalDecoderRuntime::Dense => {
                decoder.attention_scale.unwrap_or_else(|| f64::from(head_dim).sqrt().recip())
            },
            MetalDecoderRuntime::SharedRouted | MetalDecoderRuntime::DenseMixed => {
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

/// Refuses what only the dense runtime executes: residual and logit scales,
/// `LayerNorm`, post-norm and parallel blocks, score softcaps, and `LongRoPE`.
#[allow(clippy::float_cmp, reason = "the identity scales are exact constants")]
fn dense_only(spec: &SemanticModelSpec, runtime: MetalDecoderRuntime) -> Result<()> {
    let refuse =
        |what: &str| Err(ArchitectureError::invalid(format!("Metal {runtime:?} has no {what}")));
    if spec.decoder.residual_multiplier != 1.0 || spec.decoder.logit_divisor != 1.0 {
        return refuse("residual multiplier or logit divisor");
    }
    let layers = &spec.decoder.layers;
    if spec.decoder.final_norm.kind == NormalizationKind::Layer
        || layers.iter().any(|layer| layer.input_norm.kind == NormalizationKind::Layer)
    {
        return refuse("LayerNorm");
    }
    for layer in layers {
        let MixerSpec::SoftmaxAttention(attention) = &layer.mixer else {
            continue;
        };
        if !layer.norm_placement.is_pre() {
            return refuse("post-norm or parallel blocks");
        }
        if attention.logit_softcap.is_some() {
            return refuse("attention score softcap");
        }
        if let PositionEncodingSpec::Rotary(rotary) = &attention.position
            && matches!(rotary.scaling, Some(RopeScalingSpec::LongRope { .. }))
        {
            return refuse("LongRoPE rotary scaling");
        }
        if matches!(attention.window, Some(AttentionWindow::Chunked { .. })) {
            return refuse("chunked attention");
        }
        if attention.query_temperature.is_some() {
            return refuse("query temperature");
        }
    }
    Ok(())
}
