//! Dense (BF16) checkpoints with per-expert tensors, as Transformers stores
//! Qwen3-MoE and DeepSeek-V2, load and generate on Metal wherever Metal
//! admits them: admission and the expert loaders read one layout.
#![cfg(feature = "metal")]

mod fixture;

use std::io::Write;

use libmir::{
    AdmissionStatus, BackendTarget, GenerationOverrides, Library, ModelDescriptor, RuntimeConfig,
    SamplingLogits,
};

type TestResult<T> = Result<T, Box<dyn std::error::Error>>;

fn loads_what_metal_admits(name: &str, checkpoint: &fixture::Checkpoint) -> TestResult<()> {
    let root =
        std::env::temp_dir().join(format!("libmir-dense-experts-{name}-{}", std::process::id()));
    drop(std::fs::remove_dir_all(&root));
    checkpoint.write(&root)?;
    let descriptor = ModelDescriptor::inspect(&root, GenerationOverrides::default())?;
    let metal = descriptor.admission(BackendTarget::Metal);
    assert_eq!(metal.status, AdmissionStatus::Supported, "{metal:#?}");

    let mut config = RuntimeConfig::default();
    config.kv_cache.block_count = 8;
    config.automatic_kv_cache = false;
    let library = Library::new(config);
    let model = library.load(&root, GenerationOverrides::default(), &mut |_event| {})?;
    let mut session = model.session();
    let prefill = session.prefill(&[1, 2, 3, 4], SamplingLogits::Full, &mut |_event| {})?;
    let logits = prefill.logits.as_ref().ok_or("prefill returned no logits")?;
    assert_eq!(logits.values.len(), fixture::VOCAB);
    assert!(logits.values.iter().all(|value| value.is_finite()), "{:?}", logits.values);
    let decoded = session.decode(5, SamplingLogits::None)?;
    assert!(decoded.event.token_id.is_some_and(|token| (token as usize) < fixture::VOCAB));
    drop(session);
    model.unload()?;
    std::fs::remove_dir_all(root)?;
    Ok(())
}

/// Qwen3-MoE: routed experts without a shared expert, on the dense runtime.
#[test]
fn per_expert_dense_routed_experts_generate_on_metal() -> TestResult<()> {
    loads_what_metal_admits("qwen3-moe", &fixture::qwen3_moe())
}

/// DeepSeek-V2: latent attention, a leading dense layer, then routed plus
/// shared experts, on the latent-attention runtime.
#[test]
fn per_expert_dense_latent_experts_generate_on_metal() -> TestResult<()> {
    loads_what_metal_admits("deepseek-v2", &fixture::deepseek_v2())
}

/// A real per-expert checkpoint (Qwen3-30B-A3B, DeepSeek-V2-Lite in BF16)
/// loads with its expert banks taking no more than the checkpoint holds:
/// stacking copies each bank once and lets the per-expert reads go.
#[test]
#[ignore = "loads a real per-expert dense checkpoint; set LIBMIR_DENSE_EXPERTS_MODEL"]
fn a_real_per_expert_checkpoint_loads_within_its_size() -> TestResult<()> {
    let root = std::path::PathBuf::from(
        std::env::var_os("LIBMIR_DENSE_EXPERTS_MODEL").ok_or("set LIBMIR_DENSE_EXPERTS_MODEL")?,
    );
    let checkpoint = std::fs::read_dir(&root)?
        .filter_map(Result::ok)
        .filter(|entry| {
            entry.path().extension().is_some_and(|extension| extension == "safetensors")
        })
        .map(|entry| entry.metadata().map(|metadata| metadata.len()))
        .sum::<std::io::Result<u64>>()?;
    let mut config = RuntimeConfig::default();
    config.kv_cache.block_count = 8;
    config.automatic_kv_cache = false;
    let library = Library::new(config);
    let baseline = library.memory_snapshot()?;
    let model = library.load(&root, GenerationOverrides::default(), &mut |_event| {})?;
    let loaded = library.memory_snapshot()?;
    let active = loaded.active_bytes.saturating_sub(baseline.active_bytes);
    let mut session = model.session();
    let prefill = session.prefill(&[1, 2, 3, 4], SamplingLogits::Full, &mut |_event| {})?;
    let logits = prefill.logits.as_ref().ok_or("prefill returned no logits")?;
    assert!(logits.values.iter().all(|value| value.is_finite()));
    writeln!(
        std::io::stderr().lock(),
        "checkpoint {checkpoint} bytes, loaded active {active} bytes, cached {} bytes",
        loaded.cached_bytes
    )?;
    // The KV pages and workspaces come on top of the weights.
    assert!(active <= checkpoint + checkpoint / 10 + (1 << 30), "{active} > {checkpoint}");
    drop(session);
    model.unload()?;
    Ok(())
}
