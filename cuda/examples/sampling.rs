use std::{
    io::{self, Write},
    path::PathBuf,
};

use libmir_cuda::kernels::{Sampling, SamplingSpec, SamplingWorkspace};
use mircuda::{Compiler, Driver, bf16};

const VOCAB: usize = 262_144;
const ITERATIONS: u16 = 100;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let driver = Driver::initialize()?;
    let device = driver.devices()?.into_iter().next().ok_or(mircuda::Error::InvalidLaunch)?;
    let context = driver.create_context(device)?;
    let stream = context.create_stream()?;
    let pool = context.default_memory_pool()?;
    let cuda_home = std::env::var_os("CUDA_HOME")
        .map_or_else(|| PathBuf::from("/usr/local/cuda"), PathBuf::from);
    let include = cuda_home.join("include");
    let compiler =
        Compiler::with_include_paths(context.clone(), [include.clone(), include.join("cccl")])?;
    let logits = pool.allocate_zeroed::<bf16>(&stream, VOCAB)?;
    let workspace = Sampling::workspace_elements(VOCAB)?;
    let mut buffers = Buffers {
        output: pool.allocate::<u32>(&stream, 1)?,
        first: pool.allocate::<u64>(&stream, workspace)?,
        second: pool.allocate::<u64>(&stream, workspace)?,
        denominator: pool.allocate::<f32>(&stream, 1)?,
    };
    let operation = Sampling::compile(&compiler, VOCAB)?;
    let mut stdout = io::stdout().lock();
    for (name, top_k) in [("greedy", 1), ("top-k-64", 64)] {
        let spec = SamplingSpec {
            vocab: VOCAB,
            top_k,
            top_p: 0.95,
            temperature: 1.0,
            draw: 0.5,
        };
        let microseconds = profile(&context, &stream, &operation, &logits, &mut buffers, spec)?;
        writeln!(stdout, "{name}: {microseconds:.3} us")?;
    }
    Ok(())
}

struct Buffers {
    output: mircuda::DeviceBuffer<u32>,
    first: mircuda::DeviceBuffer<u64>,
    second: mircuda::DeviceBuffer<u64>,
    denominator: mircuda::DeviceBuffer<f32>,
}

fn profile(
    context: &mircuda::Context,
    stream: &mircuda::Stream,
    operation: &Sampling,
    logits: &mircuda::DeviceBuffer<bf16>,
    buffers: &mut Buffers,
    spec: SamplingSpec,
) -> libmir_cuda::Result<f64> {
    for _ in 0..5 {
        execute(operation, stream, logits, buffers, spec)?;
    }
    stream.synchronize()?;
    let started = context.create_event(true)?;
    let completed = context.create_event(true)?;
    started.record(stream)?;
    for _ in 0..ITERATIONS {
        execute(operation, stream, logits, buffers, spec)?;
    }
    completed.record(stream)?;
    completed.synchronize()?;
    Ok(f64::from(started.elapsed_ms(&completed)?) * 1_000.0 / f64::from(ITERATIONS))
}

fn execute(
    operation: &Sampling,
    stream: &mircuda::Stream,
    logits: &mircuda::DeviceBuffer<bf16>,
    buffers: &mut Buffers,
    spec: SamplingSpec,
) -> libmir_cuda::Result<()> {
    operation.execute(
        stream,
        logits,
        &mut buffers.output,
        SamplingWorkspace {
            first: &mut buffers.first,
            second: &mut buffers.second,
            denominator: &mut buffers.denominator,
        },
        spec,
    )
}
