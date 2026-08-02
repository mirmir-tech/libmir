use models::semantic::MixerSpec;

use super::fixtures::{
    AnyResult, clamped_routed, dense_and_routed, dense_contract, mixed_unsupported, shared_routed,
};
use crate::{cuda, metal};

#[test]
fn admits_every_generation_runtime_without_a_device() -> AnyResult<()> {
    let (task, dense) = dense_contract()?;
    let cases = [
        (
            dense.clone(),
            metal::MetalDecoderRuntime::Dense,
            cuda::CudaDecoderRuntime::Dense,
        ),
        (
            dense_and_routed(dense.clone()),
            metal::MetalDecoderRuntime::DenseAndRouted,
            cuda::CudaDecoderRuntime::DenseAndRouted,
        ),
        (
            shared_routed(dense.clone()),
            metal::MetalDecoderRuntime::SharedRouted,
            cuda::CudaDecoderRuntime::SharedRouted,
        ),
        (
            clamped_routed(dense),
            metal::MetalDecoderRuntime::ClampedRouted,
            cuda::CudaDecoderRuntime::ClampedRouted,
        ),
    ];

    for (semantic, metal_runtime, cuda_runtime) in cases {
        assert_eq!(
            metal::admit(&task, Some(&semantic))?,
            metal::MetalArchitecture::Generation(metal_runtime)
        );
        assert_eq!(
            cuda::admit(&task, Some(&semantic))?,
            cuda::CudaArchitecture::Generation(cuda_runtime)
        );
    }
    Ok(())
}

#[test]
fn preserves_backend_specific_window_admission() -> AnyResult<()> {
    let (task, mut semantic) = dense_contract()?;
    if let MixerSpec::SoftmaxAttention(attention) = &mut semantic.decoder.layers[0].mixer {
        attention.window = Some(128);
    }

    assert!(metal::admit(&task, Some(&semantic)).is_err());
    assert_eq!(
        cuda::admit(&task, Some(&semantic))?,
        cuda::CudaArchitecture::Generation(cuda::CudaDecoderRuntime::Dense)
    );
    Ok(())
}

#[test]
fn rejected_composition_names_the_missing_backend_runtime() -> AnyResult<()> {
    let (task, semantic) = dense_contract()?;
    let semantic = mixed_unsupported(semantic);
    let Some(metal_error) = metal::admit(&task, Some(&semantic)).err() else {
        return Err(std::io::Error::other("Metal admitted mixed runtime").into());
    };
    let Some(cuda_error) = cuda::admit(&task, Some(&semantic)).err() else {
        return Err(std::io::Error::other("CUDA admitted mixed runtime").into());
    };

    assert!(metal_error.to_string().contains("Metal has no runtime"));
    assert!(cuda_error.to_string().contains("CUDA has no runtime"));
    Ok(())
}
