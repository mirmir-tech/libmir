use std::io::{self, Write};

use libmir_cuda::kernels::{AffineGemvLaunch, AffineGemvSpec, AffineQuantizedGemv};
use mircuda::{Compiler, Context, Driver, MemoryPool, Stream, bf16};

const ITERATIONS: u16 = 1_000;

#[derive(Clone, Copy)]
struct Case {
    name: &'static str,
    input: usize,
    output: usize,
    group: usize,
    bits: usize,
}

struct Report {
    case: Case,
    device_ms: f32,
    bytes_per_operation: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let driver = Driver::initialize()?;
    let device = driver.devices()?.into_iter().next().ok_or(mircuda::Error::InvalidLaunch)?;
    let context = driver.create_context(device)?;
    let info = context.device_info()?;
    let stream = context.create_stream()?;
    let pool = context.default_memory_pool()?;
    pool.set_release_threshold(512 * 1_024 * 1_024)?;
    let compiler = Compiler::new(context.clone())?;
    let cases = [
        Case {
            name: "gemma-attention-int8",
            input: 2_816,
            output: 4_096,
            group: 64,
            bits: 8,
        },
        Case {
            name: "gemma-expert-gate-int4",
            input: 2_816,
            output: 704,
            group: 64,
            bits: 4,
        },
        Case {
            name: "gemma-expert-down-int4",
            input: 704,
            output: 2_816,
            group: 64,
            bits: 4,
        },
    ];
    let mut output = io::stdout().lock();
    writeln!(
        output,
        "device: {} (compute {}.{})",
        info.name, info.compute_capability.0, info.compute_capability.1
    )?;
    for case in cases {
        write_report(&mut output, &run(&context, &stream, &pool, &compiler, case)?)?;
    }
    Ok(())
}

fn run(
    context: &Context,
    stream: &Stream,
    pool: &MemoryPool,
    compiler: &Compiler,
    case: Case,
) -> libmir_cuda::Result<Report> {
    let values_per_word = 32 / case.bits;
    let packed = elements(case.output, case.input / values_per_word)?;
    let groups = elements(case.output, case.input / case.group)?;
    let mut input = pool.allocate_zeroed::<bf16>(stream, case.input)?;
    let mut weight = pool.allocate_zeroed::<u32>(stream, packed)?;
    let mut scales = pool.allocate_zeroed::<bf16>(stream, groups)?;
    let mut biases = pool.allocate_zeroed::<bf16>(stream, groups)?;
    let mut output = pool.allocate_zeroed::<bf16>(stream, case.output)?;
    let operation = AffineQuantizedGemv::compile(
        compiler,
        AffineGemvSpec::new(case.input, case.output, case.group, case.bits)?,
    )?;
    for _ in 0..10 {
        execute(&operation, stream, &input, &weight, &scales, &biases, &mut output)?;
    }
    stream.synchronize()?;
    let started = context.create_event(true)?;
    let completed = context.create_event(true)?;
    started.record(stream)?;
    for _ in 0..ITERATIONS {
        execute(&operation, stream, &input, &weight, &scales, &biases, &mut output)?;
    }
    completed.record(stream)?;
    completed.synchronize()?;
    let parameter_bytes = weight.bytes() + scales.bytes() + biases.bytes();
    std::hint::black_box((&mut input, &mut weight, &mut scales, &mut biases));
    Ok(Report {
        case,
        device_ms: started.elapsed_ms(&completed)?,
        bytes_per_operation: parameter_bytes + input.bytes() + output.bytes(),
    })
}

fn execute(
    operation: &AffineQuantizedGemv,
    stream: &Stream,
    input: &mircuda::DeviceBuffer<bf16>,
    weight: &mircuda::DeviceBuffer<u32>,
    scales: &mircuda::DeviceBuffer<bf16>,
    biases: &mircuda::DeviceBuffer<bf16>,
    output: &mut mircuda::DeviceBuffer<bf16>,
) -> libmir_cuda::Result<()> {
    operation.execute(
        stream,
        &mut AffineGemvLaunch {
            input,
            weight,
            scales,
            biases,
            output,
            matrix_index: 0,
        },
    )
}

fn write_report(output: &mut impl Write, report: &Report) -> io::Result<()> {
    let microseconds = f64::from(report.device_ms) * 1_000.0 / f64::from(ITERATIONS);
    let bytes = u32::try_from(report.bytes_per_operation).map_or(f64::NAN, f64::from);
    let bandwidth = bytes / (microseconds * 1_000.0);
    writeln!(
        output,
        "{}: in={} out={} bits={}, {:.3} us/op, {:.3} GB/s",
        report.case.name,
        report.case.input,
        report.case.output,
        report.case.bits,
        microseconds,
        bandwidth
    )
}

fn elements(left: usize, right: usize) -> libmir_cuda::Result<usize> {
    left.checked_mul(right)
        .ok_or(libmir_cuda::Error::InvalidQuantizedGemv("shape overflow"))
}
