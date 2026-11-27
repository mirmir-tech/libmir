use super::*;

#[test]
fn cuda_refuses_layers_in_different_layouts_that_metal_loads_one_by_one() -> crate::Result<()> {
    let layers = [uniform(&ggml(GgmlType::MXFP4)), uniform(&dense("BF16"))];
    assert_eq!(refused(&layers, &BackendTarget::Metal)?, None);
    let check = refused(&layers, &BackendTarget::Cuda)?;
    assert!(
        check.as_ref().is_some_and(|(kind, detail)| *kind == EncodingKind::Dense
            && detail.contains("one layout")
            && detail.contains("layer 1 dense banks")),
        "{check:?}"
    );
    Ok(())
}

#[test]
fn only_metal_binds_separate_block_banks_projection_by_projection() -> crate::Result<()> {
    let layers = [uniform(&block(BlockQuantization::NVFP4, TensorPacking::Separate))];
    assert_eq!(refused(&layers, &BackendTarget::Metal)?, None);
    let check = refused(&layers, &BackendTarget::Cuda)?;
    assert!(
        check.as_ref().is_some_and(
            |(kind, detail)| *kind == EncodingKind::NvFp4 && detail.contains("NVFP4 banks")
        ),
        "{check:?}"
    );
    Ok(())
}

#[test]
fn refuses_expert_banks_without_the_bias_the_loaders_add() -> crate::Result<()> {
    let plain = TensorStorage::Ggml {
        format: GgmlBlockQuantization { ggml_type: GgmlType::MXFP4 },
        companion: GgmlCompanion::Plain,
    };
    for backend in &BACKENDS {
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
