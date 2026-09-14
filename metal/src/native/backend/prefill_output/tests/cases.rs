use super::*;

#[test]
#[ignore = "real-model prefill abandonment, failed drain and worker shutdown"]
fn abandons_real_model_prefill_on_worker() -> Result<()> {
    abandonment::exercise(&manifest(std::env::var("MIRMIR_BENCH_MODEL")?))
}

#[test]
fn abandoning_last_prefill_handle_releases_completed_and_partial_rows() -> Result<()> {
    use crate::engine::clamped_routed::tests::fixture::{write_config, write_weights};
    let root = std::env::temp_dir().join(format!("prefill-abandonment-{}", Uuid::new_v4()));
    std::fs::create_dir_all(&root)?;
    write_config(&root)?;
    write_weights(&root.join("model.safetensors"), true)?;
    let result = abandonment::exercise(&manifest(root.to_string_lossy().into_owned()));
    std::fs::remove_dir_all(root)?;
    result
}

#[test]
fn abandoning_hybrid_prefill_releases_completed_and_partial_rows() -> Result<()> {
    use crate::engine::hybrid_linear_moe::tests::{write_config, write_weights};
    let root = std::env::temp_dir().join(format!("hybrid-prefill-abandonment-{}", Uuid::new_v4()));
    std::fs::create_dir_all(&root)?;
    write_config(&root)?;
    write_weights(&root.join("model.safetensors"))?;
    let result = abandonment::exercise(&manifest(root.to_string_lossy().into_owned()));
    std::fs::remove_dir_all(root)?;
    result
}

#[test]
fn materialization_failure_retires_scalar_and_complete_batch() -> Result<()> {
    use crate::engine::clamped_routed::tests::fixture::{write_config, write_weights};
    let root = std::env::temp_dir().join(format!("prefill-output-recovery-{}", Uuid::new_v4()));
    std::fs::create_dir_all(&root)?;
    write_config(&root)?;
    write_weights(&root.join("model.safetensors"), true)?;
    let result = recovery::exercise(&manifest(root.to_string_lossy().into_owned()));
    std::fs::remove_dir_all(root)?;
    result
}

#[test]
fn reports_scalar_packed_and_prefix_timings_for_clamped_moe() -> Result<()> {
    use crate::engine::clamped_routed::tests::fixture::{write_config, write_weights};
    let root = std::env::temp_dir().join(format!("clamped-prefill-timing-{}", Uuid::new_v4()));
    std::fs::create_dir_all(&root)?;
    write_config(&root)?;
    write_weights(&root.join("model.safetensors"), true)?;
    let result = exercise(&manifest(root.to_string_lossy().into_owned()));
    std::fs::remove_dir_all(root)?;
    result
}

#[test]
fn reports_scalar_packed_and_prefix_timings_for_hybrid_moe() -> Result<()> {
    use crate::engine::hybrid_linear_moe::tests::{write_config, write_weights};
    let root = std::env::temp_dir().join(format!("hybrid-prefill-timing-{}", Uuid::new_v4()));
    std::fs::create_dir_all(&root)?;
    write_config(&root)?;
    write_weights(&root.join("model.safetensors"))?;
    let result = exercise(&manifest(root.to_string_lossy().into_owned()));
    std::fs::remove_dir_all(root)?;
    result
}

#[test]
#[ignore = "real-model correctness through scalar/packed/prefix output adapters"]
fn reports_real_model_prefill_timings() -> Result<()> {
    exercise(&manifest(std::env::var("MIRMIR_BENCH_MODEL")?))
}
