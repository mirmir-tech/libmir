//! Prints accelerator pool and host memory after each model startup step:
//! load, profile warm-up, and an allocator trim. On unified-memory hosts the
//! host figure covers allocations the pool does not see, such as the CUDA
//! context, compiled modules and vendor-library workspaces.
//!
//! `memory-steps <model> [kv_blocks] [max_batch_requests] [reserve_gib]`
#![allow(clippy::print_stdout)]

use std::{env, num::NonZeroU32};

use libmir::{GenerationOverrides, Library, MemorySnapshot, ModelLoadOptions, RuntimeConfig};

const GIB: f64 = 1_073_741_824.0;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    drop(
        tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .try_init(),
    );
    let model = env::args()
        .nth(1)
        .ok_or("usage: memory-steps <model> [kv_blocks] [rows] [reserve_gib]")?;
    let kv_blocks = argument(2, 8_192)?;
    let rows = argument(3, 3)?;
    let reserve = argument(4, 32)?;
    let mut runtime = RuntimeConfig::default();
    runtime.kv_cache.block_count = u32::try_from(kv_blocks)?;
    runtime.automatic_kv_cache = false;
    runtime.scheduler.max_batch_requests = rows;
    runtime.memory.reserve_bytes = Some(u64::try_from(reserve)? << 30);
    let library = Library::new(runtime);
    let baseline = library.memory_snapshot()?;
    report("engine", &baseline, &baseline);
    let options = ModelLoadOptions {
        kv_blocks: NonZeroU32::new(u32::try_from(kv_blocks)?),
        ..ModelLoadOptions::default()
    };
    let model =
        library.load_with_options(&model, GenerationOverrides::default(), options, &mut |_| {})?;
    report("loaded", &baseline, &library.memory_snapshot()?);
    model.warm_execution_profiles(&mut |_| {})?;
    report("warm", &baseline, &library.memory_snapshot()?);
    model.engine().clear_memory_cache()?;
    report("trimmed", &baseline, &library.memory_snapshot()?);
    Ok(())
}

fn argument(index: usize, default: usize) -> Result<usize, Box<dyn std::error::Error>> {
    env::args().nth(index).map_or(Ok(default), |value| Ok(value.parse()?))
}

fn report(step: &str, baseline: &MemorySnapshot, now: &MemorySnapshot) {
    let host = baseline
        .available_bytes
        .unwrap_or_default()
        .saturating_sub(now.available_bytes.unwrap_or_default());
    let pool = now.active_bytes.saturating_add(now.cached_bytes);
    #[allow(clippy::cast_precision_loss)]
    let gib = |bytes: u64| bytes as f64 / GIB;
    println!(
        "{step:8} host_used={:.2} pool_active={:.2} pool_cached={:.2} outside_pool={:.2} GiB",
        gib(host),
        gib(now.active_bytes),
        gib(now.cached_bytes),
        gib(host.saturating_sub(pool)),
    );
}
