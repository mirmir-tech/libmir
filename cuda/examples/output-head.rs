use std::{
    env,
    io::{self, Write},
    path::PathBuf,
};

use libmir_cuda::kernels::{
    BlockFp8LinearKernels, BlockFp8LinearSpec, Fp8OutputSpec, Fp8RefinementKernels,
};
use mircuda::{
    Compiler, Context, DenseVectorPlan, DenseVectorSpec, Driver, MemoryPool, Stream, bf16,
};

const INPUT: usize = 2_880;
const OUTPUT: usize = 201_088;
const ITERATIONS: u16 = 32;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let driver = Driver::initialize()?;
    let device = driver.devices()?.into_iter().next().ok_or(mircuda::Error::InvalidLaunch)?;
    let context = driver.create_context(device)?;
    let info = context.device_info()?;
    let stream = context.create_stream()?;
    let pool = context.default_memory_pool()?;
    pool.set_release_threshold(2 * 1_024 * 1_024 * 1_024)?;
    let report = profile(&context, &stream, &pool)?;
    let mut output = io::stdout().lock();
    writeln!(
        output,
        "device: {} (compute {}.{})",
        info.name, info.compute_capability.0, info.compute_capability.1
    )?;
    writeln!(output, "GPT-OSS output head: {INPUT} -> {OUTPUT}")?;
    writeln!(output, "  exact BF16:       {:.3} us/token", report.exact)?;
    writeln!(output, "  block FP8:        {:.3} us/token", report.fp8)?;
    writeln!(output, "  refinement only:  {:.3} us/token", report.refine)?;
    writeln!(output, "  FP8 + refinement: {:.3} us/token", report.full)?;
    writeln!(output, "  exact/full speedup: {:.3}x", report.exact / report.full)?;
    Ok(())
}

struct Report {
    exact: f64,
    fp8: f64,
    refine: f64,
    full: f64,
}

fn profile(context: &Context, stream: &Stream, pool: &MemoryPool) -> libmir_cuda::Result<Report> {
    let input = pool.allocate_zeroed::<bf16>(stream, INPUT)?;
    let exact_weight = pool.allocate_zeroed::<bf16>(stream, INPUT * OUTPUT)?;
    let mut logits = pool.allocate_zeroed::<bf16>(stream, OUTPUT)?;
    let cuda_home =
        env::var_os("CUDA_HOME").map_or_else(|| PathBuf::from("/usr/local/cuda"), PathBuf::from);
    let include = cuda_home.join("include");
    let compiler =
        Compiler::with_include_paths(context.clone(), [include.clone(), include.join("cccl")])?;
    let spec = BlockFp8LinearSpec::new(INPUT, OUTPUT)?;
    let kernels = BlockFp8LinearKernels::compile(&compiler, spec)?;
    let mut weight = pool.allocate::<u8>(stream, spec.weight_elements()?)?;
    let mut scales = pool.allocate::<f32>(stream, spec.scale_elements()?)?;
    kernels.quantize(stream, &exact_weight, &mut weight, &mut scales)?;
    let refinement =
        Fp8RefinementKernels::compile(&compiler, Fp8OutputSpec::new_refinement(INPUT, OUTPUT)?)?;
    let workspace = Fp8RefinementKernels::workspace_elements(OUTPUT)?;
    let mut first = pool.allocate::<u64>(stream, workspace)?;
    let mut second = pool.allocate::<u64>(stream, workspace)?;
    let mut exact = DenseVectorPlan::new(context, stream, DenseVectorSpec::new(OUTPUT, INPUT)?)?;
    for _ in 0..4 {
        kernels.project(stream, &input, &weight, &scales, &mut logits)?;
        refinement.execute(stream, &input, &exact_weight, &mut logits, &mut first, &mut second)?;
    }
    stream.synchronize()?;
    let exact_us = time(context, stream, || {
        Ok(exact.execute(stream, &input, &exact_weight, &mut logits, 1.0, 0.0)?)
    })?;
    let fp8_us = time(context, stream, || {
        kernels.project(stream, &input, &weight, &scales, &mut logits)
    })?;
    let refine_us = time(context, stream, || {
        refinement.execute(stream, &input, &exact_weight, &mut logits, &mut first, &mut second)
    })?;
    let full_us = time(context, stream, || {
        kernels.project(stream, &input, &weight, &scales, &mut logits)?;
        refinement.execute(stream, &input, &exact_weight, &mut logits, &mut first, &mut second)
    })?;
    Ok(Report {
        exact: exact_us,
        fp8: fp8_us,
        refine: refine_us,
        full: full_us,
    })
}

fn time(
    context: &Context,
    stream: &Stream,
    mut operation: impl FnMut() -> libmir_cuda::Result<()>,
) -> libmir_cuda::Result<f64> {
    let started = context.create_event(true)?;
    let completed = context.create_event(true)?;
    started.record(stream)?;
    for _ in 0..ITERATIONS {
        operation()?;
    }
    completed.record(stream)?;
    completed.synchronize()?;
    Ok(f64::from(started.elapsed_ms(&completed)?) * 1_000.0 / f64::from(ITERATIONS))
}
