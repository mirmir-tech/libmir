use std::io::{self, Write};

use libmir_cuda::kernels::{
    AffineGemvSpec, GatedActivation, SelectedAffineGated, SelectedAffineGatedLaunch,
    SelectedAffineGatedSpec, SelectedAffineReduce, SelectedAffineReduceLaunch,
    SelectedAffineReduceSpec,
};
use mircuda::{Compiler, Context, DeviceBuffer, Driver, MemoryPool, Stream, bf16};

const HIDDEN: usize = 2_816;
const INTERMEDIATE: usize = 704;
const EXPERTS: usize = 128;
const SELECTED: usize = 8;
const GROUP: usize = 64;
const BITS: usize = 4;
const ITERATIONS: u16 = 300;

#[derive(Clone, Copy)]
struct Bank<'a> {
    weight: &'a DeviceBuffer<u32>,
    scales: &'a DeviceBuffer<bf16>,
    biases: &'a DeviceBuffer<bf16>,
}

struct Operations {
    gated: SelectedAffineGated,
    reduce: SelectedAffineReduce,
}

struct Buffers<'a> {
    input: &'a DeviceBuffer<bf16>,
    selected: &'a DeviceBuffer<u32>,
    routing: &'a DeviceBuffer<bf16>,
    gate: Bank<'a>,
    up: Bank<'a>,
    down: Bank<'a>,
    intermediate: &'a mut DeviceBuffer<bf16>,
    output: &'a mut DeviceBuffer<bf16>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let driver = Driver::initialize()?;
    let device = driver.devices()?.into_iter().next().ok_or(mircuda::Error::InvalidLaunch)?;
    let context = driver.create_context(device)?;
    let info = context.device_info()?;
    let stream = context.create_stream()?;
    let pool = context.default_memory_pool()?;
    pool.set_release_threshold(1_024 * 1_024 * 1_024)?;
    let report = profile(&context, &stream, &pool)?;
    let mut output = io::stdout().lock();
    writeln!(
        output,
        "device: {} (compute {}.{})",
        info.name, info.compute_capability.0, info.compute_capability.1
    )?;
    writeln!(output, "gemma expert MLP: top-k={SELECTED}, affine Int{BITS}, GELU tanh")?;
    writeln!(output, "  selected gate/up + activation: {:.3} us/token", report.gated_us)?;
    writeln!(output, "  selected down + reduction:     {:.3} us/token", report.reduce_us)?;
    writeln!(output, "  complete expert path:          {:.3} us/token", report.full_us)?;
    writeln!(output, "  effective parameter rate:      {:.3} GB/s", report.bandwidth)?;
    Ok(())
}

struct Report {
    gated_us: f64,
    reduce_us: f64,
    full_us: f64,
    bandwidth: f64,
}

fn profile(context: &Context, stream: &Stream, pool: &MemoryPool) -> libmir_cuda::Result<Report> {
    let mut input = pool.allocate_zeroed::<bf16>(stream, HIDDEN)?;
    let mut selected_host = context.allocate_pinned::<u32>(SELECTED)?;
    selected_host.copy_from_slice(&[0, 1, 2, 3, 4, 5, 6, 7])?;
    let mut selected = pool.allocate::<u32>(stream, SELECTED)?;
    stream.copy_to_device(&mut selected_host, &mut selected)?;
    let mut routing_host = context.allocate_pinned::<bf16>(SELECTED)?;
    routing_host.copy_from_slice(&[bf16::from_f32(0.125); SELECTED])?;
    let mut routing = pool.allocate::<bf16>(stream, SELECTED)?;
    stream.copy_to_device(&mut routing_host, &mut routing)?;
    let mut gate_storage = allocate_bank(pool, stream, HIDDEN, INTERMEDIATE)?;
    let mut up_storage = allocate_bank(pool, stream, HIDDEN, INTERMEDIATE)?;
    let mut down_storage = allocate_bank(pool, stream, INTERMEDIATE, HIDDEN)?;
    let gate = gate_storage.bank();
    let up = up_storage.bank();
    let down = down_storage.bank();
    let mut intermediate = pool.allocate_zeroed::<bf16>(stream, SELECTED * INTERMEDIATE)?;
    let mut output = pool.allocate_zeroed::<bf16>(stream, HIDDEN)?;
    let compiler = Compiler::new(context.clone())?;
    let gated_matrix = AffineGemvSpec::new(HIDDEN, INTERMEDIATE, GROUP, BITS)?;
    let down_matrix = AffineGemvSpec::new(INTERMEDIATE, HIDDEN, GROUP, BITS)?;
    let operations = Operations {
        gated: SelectedAffineGated::compile(
            &compiler,
            SelectedAffineGatedSpec::new(
                gated_matrix,
                EXPERTS,
                SELECTED,
                GatedActivation::GeluTanh,
            )?,
        )?,
        reduce: SelectedAffineReduce::compile(
            &compiler,
            SelectedAffineReduceSpec::new(down_matrix, EXPERTS, SELECTED)?,
        )?,
    };
    let mut buffers = Buffers {
        input: &input,
        selected: &selected,
        routing: &routing,
        gate,
        up,
        down,
        intermediate: &mut intermediate,
        output: &mut output,
    };
    for _ in 0..10 {
        execute_full(&operations, stream, &mut buffers)?;
    }
    stream.synchronize()?;
    let gated_us =
        time(context, stream, || execute_gated(&operations.gated, stream, &mut buffers))?;
    let reduce_us =
        time(context, stream, || execute_reduce(&operations.reduce, stream, &mut buffers))?;
    let full_us = time(context, stream, || execute_full(&operations, stream, &mut buffers))?;
    let gated_bytes = bank_bytes(HIDDEN, INTERMEDIATE)? * SELECTED * 2;
    let down_bytes = bank_bytes(INTERMEDIATE, HIDDEN)? * SELECTED;
    let bytes = u32::try_from(gated_bytes + down_bytes).map_or(f64::NAN, f64::from);
    std::hint::black_box((&mut input, &mut gate_storage, &mut up_storage, &mut down_storage));
    Ok(Report {
        gated_us,
        reduce_us,
        full_us,
        bandwidth: bytes / (full_us * 1_000.0),
    })
}

struct BankStorage {
    weight: DeviceBuffer<u32>,
    scales: DeviceBuffer<bf16>,
    biases: DeviceBuffer<bf16>,
}

impl BankStorage {
    const fn bank(&self) -> Bank<'_> {
        Bank {
            weight: &self.weight,
            scales: &self.scales,
            biases: &self.biases,
        }
    }
}

fn allocate_bank(
    pool: &MemoryPool,
    stream: &Stream,
    input: usize,
    output: usize,
) -> libmir_cuda::Result<BankStorage> {
    let packed = EXPERTS * output * input / (32 / BITS);
    let grouped = EXPERTS * output * input / GROUP;
    Ok(BankStorage {
        weight: pool.allocate_zeroed::<u32>(stream, packed)?,
        scales: pool.allocate_zeroed::<bf16>(stream, grouped)?,
        biases: pool.allocate_zeroed::<bf16>(stream, grouped)?,
    })
}

fn execute_full(
    operations: &Operations,
    stream: &Stream,
    buffers: &mut Buffers<'_>,
) -> libmir_cuda::Result<()> {
    execute_gated(&operations.gated, stream, buffers)?;
    execute_reduce(&operations.reduce, stream, buffers)
}

fn execute_gated(
    operation: &SelectedAffineGated,
    stream: &Stream,
    buffers: &mut Buffers<'_>,
) -> libmir_cuda::Result<()> {
    operation.execute(
        stream,
        &mut SelectedAffineGatedLaunch {
            input: buffers.input,
            selected: buffers.selected,
            gate_weight: buffers.gate.weight,
            gate_scales: buffers.gate.scales,
            gate_biases: buffers.gate.biases,
            up_weight: buffers.up.weight,
            up_scales: buffers.up.scales,
            up_biases: buffers.up.biases,
            output: &mut *buffers.intermediate,
        },
    )
}

fn execute_reduce(
    operation: &SelectedAffineReduce,
    stream: &Stream,
    buffers: &mut Buffers<'_>,
) -> libmir_cuda::Result<()> {
    operation.execute(
        stream,
        &mut SelectedAffineReduceLaunch {
            input: &*buffers.intermediate,
            selected: buffers.selected,
            routing_weights: buffers.routing,
            weight: buffers.down.weight,
            scales: buffers.down.scales,
            biases: buffers.down.biases,
            output: &mut *buffers.output,
        },
    )
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

fn bank_bytes(input: usize, output: usize) -> libmir_cuda::Result<usize> {
    let packed = output * input / (32 / BITS) * size_of::<u32>();
    let grouped = output * input / GROUP * size_of::<bf16>() * 2;
    packed
        .checked_add(grouped)
        .ok_or(libmir_cuda::Error::InvalidQuantizedGemv("shape overflow"))
}
