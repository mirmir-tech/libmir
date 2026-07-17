use std::{
    io::{self, Write},
    path::PathBuf,
};

use libmir_cuda::kernels::{AffineGemvSpec, AffineQmmLaunch, AffineQmmSpec, AffineQuantizedQmm};
use mircuda::{Compiler, Context, Driver, MemoryPool, Stream, bf16};

const ITERATIONS: u16 = 50;

#[derive(Clone, Copy)]
struct Case {
    name: &'static str,
    tokens: usize,
    input: usize,
    output: usize,
    group: usize,
    bits: usize,
}

const CASES: [Case; 6] = [
    Case {
        name: "attention-short",
        tokens: 8,
        input: 2_816,
        output: 4_096,
        group: 64,
        bits: 8,
    },
    Case {
        name: "attention-chunk",
        tokens: 64,
        input: 2_816,
        output: 4_096,
        group: 64,
        bits: 8,
    },
    Case {
        name: "attention-prefill",
        tokens: 512,
        input: 2_816,
        output: 4_096,
        group: 64,
        bits: 8,
    },
    Case {
        name: "expert-short",
        tokens: 8,
        input: 2_816,
        output: 704,
        group: 64,
        bits: 4,
    },
    Case {
        name: "expert-chunk",
        tokens: 64,
        input: 2_816,
        output: 704,
        group: 64,
        bits: 4,
    },
    Case {
        name: "expert-prefill",
        tokens: 512,
        input: 2_816,
        output: 704,
        group: 64,
        bits: 4,
    },
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let driver = Driver::initialize()?;
    let device = driver.devices()?.into_iter().next().ok_or(mircuda::Error::InvalidLaunch)?;
    let context = driver.create_context(device)?;
    let info = context.device_info()?;
    let stream = context.create_stream()?;
    let pool = context.default_memory_pool()?;
    pool.set_release_threshold(1_024 * 1_024 * 1_024)?;
    let compiler =
        Compiler::with_include_paths(context.clone(), [PathBuf::from("/usr/local/cuda/include")])?;
    let mut output = io::stdout().lock();
    writeln!(
        output,
        "device: {} (compute {}.{})",
        info.name, info.compute_capability.0, info.compute_capability.1
    )?;
    for case in CASES {
        let report = profile(&context, &stream, &pool, &compiler, case)?;
        writeln!(
            output,
            "{}: tokens={} in={} out={} Int{}, {:.3} us, {:.0} token/s, {:.3} TFLOP/s",
            case.name,
            case.tokens,
            case.input,
            case.output,
            case.bits,
            report.microseconds,
            report.tokens_per_second,
            report.teraflops,
        )?;
    }
    Ok(())
}

struct Report {
    microseconds: f64,
    tokens_per_second: f64,
    teraflops: f64,
}

#[derive(Clone, Copy)]
struct Buffers<'a> {
    input: &'a mircuda::DeviceBuffer<bf16>,
    weight: &'a mircuda::DeviceBuffer<u32>,
    scales: &'a mircuda::DeviceBuffer<bf16>,
    biases: &'a mircuda::DeviceBuffer<bf16>,
}

fn profile(
    context: &Context,
    stream: &Stream,
    pool: &MemoryPool,
    compiler: &Compiler,
    case: Case,
) -> libmir_cuda::Result<Report> {
    let values_per_word = 32 / case.bits;
    let packed = elements(case.output, case.input / values_per_word)?;
    let grouped = elements(case.output, case.input / case.group)?;
    let mut input = pool.allocate_zeroed::<bf16>(stream, elements(case.tokens, case.input)?)?;
    let mut weight = pool.allocate_zeroed::<u32>(stream, packed)?;
    let mut scales = pool.allocate_zeroed::<bf16>(stream, grouped)?;
    let mut biases = pool.allocate_zeroed::<bf16>(stream, grouped)?;
    let mut output = pool.allocate_zeroed::<bf16>(stream, elements(case.tokens, case.output)?)?;
    let matrix = AffineGemvSpec::new(case.input, case.output, case.group, case.bits)?;
    let operation =
        AffineQuantizedQmm::compile(compiler, AffineQmmSpec::new(matrix, case.tokens)?)?;
    let buffers = Buffers {
        input: &input,
        weight: &weight,
        scales: &scales,
        biases: &biases,
    };
    for _ in 0..5 {
        execute(&operation, stream, buffers, &mut output)?;
    }
    stream.synchronize()?;
    let started = context.create_event(true)?;
    let completed = context.create_event(true)?;
    started.record(stream)?;
    for _ in 0..ITERATIONS {
        execute(&operation, stream, buffers, &mut output)?;
    }
    completed.record(stream)?;
    completed.synchronize()?;
    let microseconds = f64::from(started.elapsed_ms(&completed)?) * 1_000.0 / f64::from(ITERATIONS);
    let tokens = f64::from(u32::try_from(case.tokens)?);
    let operations = 2.0
        * tokens
        * f64::from(u32::try_from(case.input)?)
        * f64::from(u32::try_from(case.output)?);
    std::hint::black_box((&mut input, &mut weight, &mut scales, &mut biases));
    Ok(Report {
        microseconds,
        tokens_per_second: tokens * 1_000_000.0 / microseconds,
        teraflops: operations / (microseconds * 1_000_000.0),
    })
}

fn execute(
    operation: &AffineQuantizedQmm,
    stream: &Stream,
    buffers: Buffers<'_>,
    output: &mut mircuda::DeviceBuffer<bf16>,
) -> libmir_cuda::Result<()> {
    operation.execute(
        stream,
        &mut AffineQmmLaunch {
            input: buffers.input,
            weight: buffers.weight,
            scales: buffers.scales,
            biases: buffers.biases,
            output,
            matrix_index: 0,
        },
    )
}

fn elements(left: usize, right: usize) -> libmir_cuda::Result<usize> {
    left.checked_mul(right)
        .ok_or(libmir_cuda::Error::InvalidQuantizedGemv("shape overflow"))
}
