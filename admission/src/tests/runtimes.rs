use models::semantic::{KeyValueRelation, MixerSpec, QkNormalization};

use super::fixtures::{AnyResult, dense_contract, mixed_unsupported, unshared_routed};
use crate::{
    ArchitectureError, Runtime,
    capability::{FeedForwardShape, LayerShape, MixerKind, Requirement},
    cuda, metal,
};

#[test]
fn admits_windowed_dense_layers_on_both_backends() -> AnyResult<()> {
    let (task, mut semantic) = dense_contract()?;
    if let MixerSpec::SoftmaxAttention(attention) = &mut semantic.decoder.layers[0].mixer {
        attention.window =
            std::num::NonZeroUsize::new(128).map(models::semantic::AttentionWindow::Sliding);
    }
    assert_eq!(
        metal::admit(&task, Some(&semantic))?.runtime(),
        Some(metal::MetalDecoderRuntime::Dense)
    );
    assert_eq!(
        cuda::admit(&task, Some(&semantic))?.runtime(),
        Some(cuda::CudaDecoderRuntime::Dense)
    );
    Ok(())
}

/// Routed experts behind attention with sinks need the clamped runtime's
/// attention, which runs only clamped experts.
#[test]
fn cuda_refuses_unclamped_experts_behind_sinks() -> AnyResult<()> {
    let (task, dense) = dense_contract()?;
    let mut sinks = unshared_routed(dense);
    for layer in &mut sinks.decoder.layers {
        if let MixerSpec::SoftmaxAttention(attention) = &mut layer.mixer {
            attention.sinks = true;
        }
    }
    let Some(ArchitectureError::NoRuntime { rejections }) = cuda::admit(&task, Some(&sinks)).err()
    else {
        return Err("CUDA admitted unclamped experts behind sinks".into());
    };
    let dense = rejections
        .iter()
        .find(|rejection| rejection.runtime == Runtime::Cuda(cuda::CudaDecoderRuntime::Dense));
    let layer = Requirement::Layer(LayerShape {
        mixer: MixerKind::SoftmaxSinks,
        feed_forward: FeedForwardShape::Routed,
    });
    assert!(dense.is_some_and(|rejection| rejection.missing.contains(&layer)));
    Ok(())
}

/// A refusal lists every runtime of the backend with what it lacks.
#[test]
fn rejected_composition_names_what_each_runtime_lacks() -> AnyResult<()> {
    let (task, semantic) = dense_contract()?;
    let semantic = mixed_unsupported(semantic);
    let Some(ArchitectureError::NoRuntime { rejections }) =
        metal::admit(&task, Some(&semantic)).err()
    else {
        return Err("Metal admitted a mixed composition".into());
    };
    assert_eq!(rejections.len(), metal::MetalDecoderRuntime::ALL.len());
    assert!(rejections.iter().all(|rejection| !rejection.missing.is_empty()));
    let Some(ArchitectureError::NoRuntime { rejections }) =
        cuda::admit(&task, Some(&semantic)).err()
    else {
        return Err("CUDA admitted a mixed composition".into());
    };
    assert_eq!(rejections.len(), cuda::CudaDecoderRuntime::ALL.len());
    Ok(())
}

/// Each layer keeps its own Q/K/V normalization (Llama 4 normalizes its
/// rotary layers only), and layers whose keys double as values say so.
#[test]
fn graph_decoder_keeps_each_layer_normalization() -> AnyResult<()> {
    let (task, mut semantic) = dense_contract()?;
    for (index, layer) in semantic.decoder.layers.iter_mut().enumerate() {
        if let MixerSpec::SoftmaxAttention(attention) = &mut layer.mixer {
            attention.qk_normalization = if index == 0 {
                attention.key_value_relation = KeyValueRelation::KeyEqualsValue;
                QkNormalization::QueryKeyRms
            } else {
                QkNormalization::QueryKeyValueRms
            };
        }
    }
    let plan = cuda::CudaDecoderPlan::lower(&semantic, None)?;
    let normalizations = plan.layers().iter().map(|layer| layer.mixer.normalization());
    assert_eq!(
        normalizations.collect::<Vec<_>>(),
        [Some(QkNormalization::QueryKeyRms), Some(QkNormalization::QueryKeyValueRms)]
    );
    assert_eq!(
        cuda::admit(&task, Some(&semantic))?.runtime(),
        Some(cuda::CudaDecoderRuntime::Dense)
    );
    Ok(())
}
