use super::*;

#[test]
fn cuda_refuses_layers_in_different_layouts_that_metal_loads_one_by_one() -> TestResult<()> {
    let layers = [uniform(&ggml(GgmlType::MXFP4)), uniform(&dense("BF16"))];
    assert_eq!(refused(&layers, BackendTarget::Metal)?, None);
    let check = refused(&layers, BackendTarget::Cuda)?;
    assert!(
        check.as_ref().is_some_and(|(kind, detail)| *kind == EncodingKind::Dense
            && detail.contains("one layout")
            && detail.contains("layer 1 dense banks")),
        "{check:?}"
    );
    Ok(())
}

#[test]
fn only_metal_binds_separate_block_banks_projection_by_projection() -> TestResult<()> {
    let layers = [uniform(&block(BlockQuantization::NVFP4, TensorPacking::Separate))];
    assert_eq!(refused(&layers, BackendTarget::Metal)?, None);
    let check = refused(&layers, BackendTarget::Cuda)?;
    assert!(
        check.as_ref().is_some_and(
            |(kind, detail)| *kind == EncodingKind::NvFp4 && detail.contains("NVFP4 banks")
        ),
        "{check:?}"
    );
    Ok(())
}

#[test]
fn refuses_expert_banks_without_the_bias_the_loaders_add() -> TestResult<()> {
    let plain = TensorStorage::Ggml {
        format: GgmlBlockQuantization { ggml_type: GgmlType::MXFP4 },
        companion: GgmlCompanion::Plain,
    };
    for backend in BACKENDS {
        let check = refused(&[uniform(&plain)], backend)?;
        assert!(
            check
                .as_ref()
                .is_some_and(|(_, detail)| detail.contains("lack the output bias")),
            "{backend:?}: {check:?}"
        );
    }
    Ok(())
}

/// GPT-OSS `SafeTensors` with MXFP4 gate and up banks and a BF16 down bank:
/// Metal's loader binds each projection on its own and loads the mix, so
/// only CUDA refuses it; a GGML mix stays refused on both.
#[test]
fn metal_admits_dense_and_block_banks_mixed_within_a_layer() -> TestResult<()> {
    let mxfp4 = block(BlockQuantization::MXFP4, TensorPacking::Separate);
    let mixed = separate(&mxfp4, &mxfp4, &dense("BF16"));
    let metal = refused(&[mixed.clone(), mixed.clone()], BackendTarget::Metal)?;
    assert!(metal.is_none(), "{metal:?}");
    let cuda = refused(&[mixed.clone(), mixed], BackendTarget::Cuda)?;
    assert!(cuda.is_some_and(|(_, detail)| detail.contains("mixes")), "CUDA reads no mix");
    let ggml = ggml(GgmlType::MXFP4);
    let ggml_mix = separate(&ggml, &ggml, &dense("BF16"));
    assert!(refused(&[ggml_mix], BackendTarget::Metal)?.is_some());
    Ok(())
}
