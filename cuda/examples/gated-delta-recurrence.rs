use std::{
    env,
    io::{self, Write},
    path::PathBuf,
};

use libmir_cuda::kernels::{
    GatedDeltaChunked, GatedDeltaChunkedScratch, GatedDeltaLaunch, GatedDeltaRecurrence,
    GatedDeltaRecurrenceMode, GatedDeltaSpec,
};
use mircuda::{Compiler, Context, DeviceBuffer, Driver, MemoryPool, Stream, bf16};

const KEY_HEADS: usize = 16;
const VALUE_HEADS: usize = 32;
const KEY_DIM: usize = 128;
const VALUE_DIM: usize = 128;
const WARMUPS: usize = 4;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tokens = argument(1, 2_048)?;
    let iterations = env::args().nth(2).map_or(Ok(20_u16), |value| value.parse())?;
    let driver = Driver::initialize()?;
    let device = driver.devices()?.into_iter().next().ok_or(mircuda::Error::InvalidLaunch)?;
    let context = driver.create_context(device)?;
    let stream = context.create_stream()?;
    let pool = context.default_memory_pool()?;
    let compiler = compiler(&context)?;
    let spec = GatedDeltaSpec {
        tokens,
        key_heads: KEY_HEADS,
        value_heads: VALUE_HEADS,
        key_dim: KEY_DIM,
        value_dim: VALUE_DIM,
    };
    let operation = GatedDeltaRecurrence::compile(&compiler, spec)?;
    let chunked = GatedDeltaChunked::compile(&compiler, spec)?;
    let serial = profile(
        &context,
        &stream,
        &operation,
        Buffers::new(&pool, &stream, spec)?,
        GatedDeltaRecurrenceMode::Serial,
        iterations,
    )?;
    let tiled_2 = profile_mode(
        &context,
        &stream,
        &pool,
        &operation,
        spec,
        GatedDeltaRecurrenceMode::ValueTiled2,
        iterations,
    )?;
    let tiled_4 = profile_mode(
        &context,
        &stream,
        &pool,
        &operation,
        spec,
        GatedDeltaRecurrenceMode::ValueTiled4,
        iterations,
    )?;
    let tiled_8 = profile_mode(
        &context,
        &stream,
        &pool,
        &operation,
        spec,
        GatedDeltaRecurrenceMode::ValueTiled8,
        iterations,
    )?;
    let chunked = profile_chunked(&context, &stream, &pool, &chunked, spec, iterations)?;
    let mut output = io::stdout().lock();
    writeln!(
        output,
        "tokens={tokens} shape=kh{KEY_HEADS}x{KEY_DIM}/vh{VALUE_HEADS}x{VALUE_DIM}"
    )?;
    writeln!(output, "serial_us={serial:.3}")?;
    writeln!(output, "value_tiled_2_us={tiled_2:.3} speedup={:.3}x", serial / tiled_2)?;
    writeln!(output, "value_tiled_4_us={tiled_4:.3} speedup={:.3}x", serial / tiled_4)?;
    writeln!(output, "value_tiled_8_us={tiled_8:.3} speedup={:.3}x", serial / tiled_8)?;
    writeln!(output, "chunked_us={chunked:.3} speedup={:.3}x", serial / chunked)?;
    Ok(())
}

fn profile_chunked(
    context: &Context,
    stream: &Stream,
    pool: &MemoryPool,
    operation: &GatedDeltaChunked,
    spec: GatedDeltaSpec,
    iterations: u16,
) -> libmir_cuda::Result<f64> {
    let mut buffers = Buffers::new(pool, stream, spec)?;
    let mut scratch = GatedDeltaChunkedScratch::new(context, pool, stream, spec)?;
    for _ in 0..WARMUPS {
        operation.execute(stream, &mut buffers.launch(), &mut scratch)?;
    }
    stream.synchronize()?;
    let started = context.create_event(true)?;
    let completed = context.create_event(true)?;
    started.record(stream)?;
    for _ in 0..iterations {
        operation.execute(stream, &mut buffers.launch(), &mut scratch)?;
    }
    completed.record(stream)?;
    completed.synchronize()?;
    Ok(f64::from(started.elapsed_ms(&completed)?) * 1_000.0 / f64::from(iterations))
}

fn profile_mode(
    context: &Context,
    stream: &Stream,
    pool: &MemoryPool,
    operation: &GatedDeltaRecurrence,
    spec: GatedDeltaSpec,
    mode: GatedDeltaRecurrenceMode,
    iterations: u16,
) -> libmir_cuda::Result<f64> {
    profile(context, stream, operation, Buffers::new(pool, stream, spec)?, mode, iterations)
}

fn profile(
    context: &Context,
    stream: &Stream,
    operation: &GatedDeltaRecurrence,
    mut buffers: Buffers,
    mode: GatedDeltaRecurrenceMode,
    iterations: u16,
) -> libmir_cuda::Result<f64> {
    for _ in 0..WARMUPS {
        operation.execute_with(stream, &mut buffers.launch(), mode)?;
    }
    stream.synchronize()?;
    let started = context.create_event(true)?;
    let completed = context.create_event(true)?;
    started.record(stream)?;
    for _ in 0..iterations {
        operation.execute_with(stream, &mut buffers.launch(), mode)?;
    }
    completed.record(stream)?;
    completed.synchronize()?;
    Ok(f64::from(started.elapsed_ms(&completed)?) * 1_000.0 / f64::from(iterations))
}

struct Buffers {
    query: DeviceBuffer<bf16>,
    key: DeviceBuffer<bf16>,
    value: DeviceBuffer<bf16>,
    alpha: DeviceBuffer<bf16>,
    beta: DeviceBuffer<bf16>,
    a_log: DeviceBuffer<bf16>,
    dt_bias: DeviceBuffer<bf16>,
    decay: DeviceBuffer<f32>,
    update: DeviceBuffer<f32>,
    state: DeviceBuffer<f32>,
    output: DeviceBuffer<bf16>,
}

impl Buffers {
    fn new(pool: &MemoryPool, stream: &Stream, spec: GatedDeltaSpec) -> libmir_cuda::Result<Self> {
        let key = spec.tokens * spec.key_heads * spec.key_dim;
        let value = spec.tokens * spec.value_heads * spec.value_dim;
        let gates = spec.tokens * spec.value_heads;
        let state = spec.value_heads * spec.value_dim * spec.key_dim;
        Ok(Self {
            query: pool.allocate_zeroed(stream, key)?,
            key: pool.allocate_zeroed(stream, key)?,
            value: pool.allocate_zeroed(stream, value)?,
            alpha: pool.allocate_zeroed(stream, gates)?,
            beta: pool.allocate_zeroed(stream, gates)?,
            a_log: pool.allocate_zeroed(stream, spec.value_heads)?,
            dt_bias: pool.allocate_zeroed(stream, spec.value_heads)?,
            decay: pool.allocate(stream, gates)?,
            update: pool.allocate(stream, gates)?,
            state: pool.allocate_zeroed(stream, state)?,
            output: pool.allocate(stream, value)?,
        })
    }

    fn launch(&mut self) -> GatedDeltaLaunch<'_> {
        GatedDeltaLaunch {
            query: &self.query,
            key: &self.key,
            value: &self.value,
            alpha: &self.alpha,
            beta: &self.beta,
            a_log: &self.a_log,
            dt_bias: &self.dt_bias,
            decay: &mut self.decay,
            update: &mut self.update,
            state: &mut self.state,
            output: &mut self.output,
        }
    }
}

fn compiler(context: &Context) -> mircuda::Result<Compiler> {
    let cuda_home =
        env::var_os("CUDA_HOME").map_or_else(|| PathBuf::from("/usr/local/cuda"), PathBuf::from);
    let include = cuda_home.join("include");
    Compiler::with_include_paths(context.clone(), [include.clone(), include.join("cccl")])
}

fn argument(index: usize, default: usize) -> Result<usize, Box<dyn std::error::Error>> {
    Ok(env::args().nth(index).map_or(Ok(default), |value| value.parse())?)
}
