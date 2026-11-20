//! A real multi-head latent attention checkpoint (DeepSeek-V2-Lite-Chat) is
//! described completely, every tensor bound, and then refused by name until a
//! backend runs latent attention. Moonlight needs a tiktoken tokenizer first.

use libmir::{
    AdmissionCheckKind, AdmissionStatus, BackendTarget, Error, GenerationOverrides,
    ModelDescriptor, Result,
};

#[test]
#[ignore = "inspects a real latent-attention checkpoint; set LIBMIR_LATENT_MODEL"]
fn binds_a_latent_checkpoint_and_refuses_it_by_name() -> Result<()> {
    let path = std::env::var_os("LIBMIR_LATENT_MODEL")
        .ok_or(Error::MissingEnvironment("LIBMIR_LATENT_MODEL"))?;
    let descriptor = ModelDescriptor::inspect(path, GenerationOverrides::default())?;

    for backend in [BackendTarget::Cuda, BackendTarget::Metal] {
        let report = descriptor.admission(backend);
        assert_eq!(report.status, AdmissionStatus::Unsupported, "{report:#?}");
        let architecture = report
            .checks
            .iter()
            .find(|check| check.kind == AdmissionCheckKind::Architecture)
            .map(|check| check.detail.as_str())
            .unwrap_or_default();
        assert!(architecture.contains("multi-head latent attention"), "{report:#?}");
    }
    Ok(())
}
