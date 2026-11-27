//! The reviewer's case on a real GPT-OSS MXFP4 GGUF's headers: admitted as
//! published, and refused on both accelerators once only the first layer's
//! down bank is dense beside GGML MXFP4 gate and up banks.

use super::*;

#[test]
#[ignore = "inspects a real GPT-OSS MXFP4 GGUF; set LIBMIR_GPT_OSS_GGUF"]
fn refuses_a_gpt_oss_gguf_layer_with_a_dense_down_bank() -> Result<()> {
    let path = std::env::var_os("LIBMIR_GPT_OSS_GGUF")
        .ok_or(Error::MissingEnvironment("LIBMIR_GPT_OSS_GGUF"))?;
    let model = models::gguf::GgufModel::open(PathBuf::from(path))?;
    let config = models::gguf::Import::derive(&model)?.config;
    let mut catalog = model.mapped_catalog()?;
    let published = RemoteModelContract::inspect_gguf(&model)?;
    let down = "model.layers.0.mlp.experts.down_proj.weight";
    let tensor = catalog
        .tensors
        .iter_mut()
        .find(|tensor| tensor.name == down)
        .ok_or_else(|| Error::from(models::ModelsError::InvalidConfig(format!("no {down}"))))?;
    tensor.dtype = "F32".into();
    let mixed = RemoteModelContract::inspect_generation(&config, &catalog)?;
    for backend in [BackendTarget::Metal, BackendTarget::Cuda] {
        let report = published.admission(backend.clone());
        assert_ne!(report.status, AdmissionStatus::Unsupported, "{report:#?}");
        let report = mixed.admission(backend);
        assert_eq!(report.status, AdmissionStatus::Unsupported, "{report:#?}");
        assert!(
            report.checks.iter().any(|check| check.detail.contains("layer 0 mixes")
                && check.detail.contains("down Dense F32")),
            "{report:#?}"
        );
    }
    Ok(())
}
