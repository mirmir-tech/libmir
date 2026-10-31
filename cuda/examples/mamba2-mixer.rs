//! Profiles the Mamba-2 mixer kernels (convolution, selective scan, gated
//! norm) at the Nemotron-3.5 Super geometry for one chunk of `tokens`.

use std::{
    env,
    io::{self, Write},
    path::PathBuf,
};

use libmir_cuda::kernels::{
    MAMBA2_CHUNK, Mamba2Geometry, Mamba2Kernels, Mamba2Launch, Mamba2Parameters,
};
use mircuda::{Compiler, Context, DeviceBuffer, Driver, MemoryPool, Stream, bf16};

const GEOMETRY: Mamba2Geometry = Mamba2Geometry {
    heads: 128,
    head_dim: 64,
    state_size: 128,
    groups: 8,
    convolution_kernel_size: 4,
    time_step_minimum: 0.0,
    time_step_maximum: f32::INFINITY,
    norm_epsilon: 1e-5,
};
const WARMUPS: usize = 3;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Path {
    Sequential,
    Chunked,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let iterations = env::args().nth(1).map_or(Ok(10_u16), |value| value.parse())?;
    let driver = Driver::initialize()?;
    let device = driver.devices()?.into_iter().next().ok_or(mircuda::Error::InvalidLaunch)?;
    let context = driver.create_context(device)?;
    let stream = context.create_stream()?;
    let pool = context.default_memory_pool()?;
    let kernels = Mamba2Kernels::compile(&compiler(&context)?, GEOMETRY)?;
    let mut output = io::stdout().lock();
    for tokens in [1, 16, 128, 512, 2048] {
        for chunked in [Path::Sequential, Path::Chunked] {
            if chunked != Path::Sequential && tokens < MAMBA2_CHUNK {
                continue;
            }
            let microseconds =
                profile(&context, &stream, &pool, &kernels, tokens, chunked, iterations)?;
            writeln!(
                output,
                "tokens={tokens} path={} mixer_us={microseconds:.1} per_token_us={:.2}",
                match chunked {
                    Path::Sequential => "sequential",
                    Path::Chunked => "chunked",
                },
                microseconds / f64::from(u32::try_from(tokens)?)
            )?;
        }
    }
    Ok(())
}

fn profile(
    context: &Context,
    stream: &Stream,
    pool: &MemoryPool,
    kernels: &Mamba2Kernels,
    tokens: usize,
    chunked: Path,
    iterations: u16,
) -> libmir_cuda::Result<f64> {
    let mut buffers = Buffers::new(pool, stream, tokens)?;
    for _ in 0..WARMUPS {
        buffers.execute(kernels, stream, tokens, chunked)?;
    }
    stream.synchronize()?;
    let started = context.create_event(true)?;
    let completed = context.create_event(true)?;
    started.record(stream)?;
    for _ in 0..iterations {
        buffers.execute(kernels, stream, tokens, chunked)?;
    }
    completed.record(stream)?;
    completed.synchronize()?;
    Ok(f64::from(started.elapsed_ms(&completed)?) * 1_000.0 / f64::from(iterations))
}

struct Buffers {
    projection: DeviceBuffer<bf16>,
    weight: DeviceBuffer<bf16>,
    bias: DeviceBuffer<bf16>,
    heads: DeviceBuffer<bf16>,
    norm: DeviceBuffer<bf16>,
    state: DeviceBuffer<f32>,
    history: DeviceBuffer<bf16>,
    next_history: DeviceBuffer<bf16>,
    convolved: DeviceBuffer<bf16>,
    scanned: DeviceBuffer<bf16>,
    output: DeviceBuffer<bf16>,
}

impl Buffers {
    fn new(pool: &MemoryPool, stream: &Stream, tokens: usize) -> libmir_cuda::Result<Self> {
        let inner = GEOMETRY.inner()?;
        let channels = GEOMETRY.convolution_channels()?;
        Ok(Self {
            projection: pool.allocate_zeroed(stream, tokens * GEOMETRY.projection_width()?)?,
            weight: pool.allocate_zeroed(stream, channels * GEOMETRY.convolution_kernel_size)?,
            bias: pool.allocate_zeroed(stream, channels)?,
            heads: pool.allocate_zeroed(stream, GEOMETRY.heads)?,
            norm: pool.allocate_zeroed(stream, inner)?,
            state: pool.allocate_zeroed(stream, GEOMETRY.state_elements()?)?,
            history: pool.allocate_zeroed(stream, GEOMETRY.history_elements()?)?,
            next_history: pool.allocate_zeroed(stream, GEOMETRY.history_elements()?)?,
            convolved: pool.allocate(stream, tokens * channels)?,
            scanned: pool.allocate(stream, tokens * inner)?,
            output: pool.allocate(stream, tokens * inner)?,
        })
    }

    fn execute(
        &mut self,
        kernels: &Mamba2Kernels,
        stream: &Stream,
        tokens: usize,
        chunked: Path,
    ) -> libmir_cuda::Result<()> {
        let mut launch = Mamba2Launch {
            tokens,
            projection: &self.projection,
            parameters: Mamba2Parameters {
                convolution_weight: &self.weight,
                convolution_bias: Some(&self.bias),
                a_log: &self.heads,
                skip: &self.heads,
                dt_bias: &self.heads,
                norm_weight: &self.norm,
            },
            state: &mut self.state,
            history: &self.history,
            next_history: &mut self.next_history,
            convolved: &mut self.convolved,
            scanned: &mut self.scanned,
            output: &mut self.output,
        };
        match chunked {
            Path::Chunked => kernels.execute_chunked(stream, &mut launch),
            Path::Sequential => kernels.execute(stream, &mut launch),
        }
    }
}

fn compiler(context: &Context) -> mircuda::Result<Compiler> {
    let cuda_home =
        env::var_os("CUDA_HOME").map_or_else(|| PathBuf::from("/usr/local/cuda"), PathBuf::from);
    let include = cuda_home.join("include");
    Compiler::with_include_paths(context.clone(), [include.clone(), include.join("cccl")])
}
