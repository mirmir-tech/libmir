//! A real multi-head latent attention checkpoint (DeepSeek-V2-Lite-Chat) is
//! described completely, every tensor bound, admitted on CUDA and refused by
//! name on Metal until it runs latent attention. Moonlight needs a tiktoken
//! tokenizer first.

use libmir::{
    AdmissionCheckKind, AdmissionStatus, BackendTarget, Error, GenerationOverrides,
    ModelDescriptor, Result,
};

#[test]
#[ignore = "inspects a real latent-attention checkpoint; set LIBMIR_LATENT_MODEL"]
fn binds_a_latent_checkpoint_and_admits_it_on_cuda() -> Result<()> {
    let path = std::env::var_os("LIBMIR_LATENT_MODEL")
        .ok_or(Error::MissingEnvironment("LIBMIR_LATENT_MODEL"))?;
    let descriptor = ModelDescriptor::inspect(path, GenerationOverrides::default())?;

    let cuda = descriptor.admission(BackendTarget::Cuda);
    assert_eq!(cuda.status, AdmissionStatus::Supported, "{cuda:#?}");

    let metal = descriptor.admission(BackendTarget::Metal);
    assert_eq!(metal.status, AdmissionStatus::Unsupported, "{metal:#?}");
    let architecture = metal
        .checks
        .iter()
        .find(|check| check.kind == AdmissionCheckKind::Architecture)
        .map(|check| check.detail.as_str())
        .unwrap_or_default();
    assert!(architecture.contains("multi-head latent attention"), "{metal:#?}");
    Ok(())
}
