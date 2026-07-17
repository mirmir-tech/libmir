use std::io::{self, Write};

use libmir_cuda::kernels::{
    AffineGemvLaunch, AffineGemvSpec, AffineQuantizedGemv, SelectedAffinePair,
    SelectedAffinePairLaunch, SelectedAffinePairSpec,
};
use mircuda::{Compiler, Context, DeviceBuffer, Driver, MemoryPool, Stream, bf16};

const INPUT: usize = 2_816;
const OUTPUT: usize = 704;
const EXPERTS: usize = 128;
const SELECTED: usize = 8;
const GROUP: usize = 64;
const BITS: usize = 4;
const ITERATIONS: u16 = 500;

#[derive(Clone, Copy)]
struct Bank<'a> {
    weight: &'a DeviceBuffer<u32>,
    scales: &'a DeviceBuffer<bf16>,
    biases: &'a DeviceBuffer<bf16>,
}

struct PairOutputs<'a> {
    gate: &'a mut DeviceBuffer<bf16>,
    up: &'a mut DeviceBuffer<bf16>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let driver = Driver::initialize()?;
    let device = driver.devices()?.into_iter().next().ok_or(mircuda::Error::InvalidLaunch)?;
    let context = driver.create_context(device)?;
    let info = context.device_info()?;
    let stream = context.create_stream()?;
    let pool = context.default_memory_pool()?;
    pool.set_release_threshold(512 * 1_024 * 1_024)?;
    let report = profile(&context, &stream, &pool)?;
    let mut output = io::stdout().lock();
    writeln!(
        output,
        "device: {} (compute {}.{})",
        info.name, info.compute_capability.0, info.compute_capability.1
    )?;
    writeln!(output, "gemma selected experts: top-k={SELECTED}, gate+up Int{BITS}")?;
    writeln!(output, "  sequential 16 launches: {:.3} us/token", report.sequential_us)?;
    writeln!(output, "  fused selected pair:    {:.3} us/token", report.fused_us)?;
    writeln!(
        output,
        "  speedup:                 {:.3}x",
        report.sequential_us / report.fused_us
    )?;
    writeln!(output, "  fused parameter rate:    {:.3} GB/s", report.bandwidth)?;
    Ok(())
}

struct Report {
    sequential_us: f64,
    fused_us: f64,
    bandwidth: f64,
}

fn profile(context: &Context, stream: &Stream, pool: &MemoryPool) -> libmir_cuda::Result<Report> {
    let values_per_word = 32 / BITS;
    let packed_per_matrix = elements(OUTPUT, INPUT / values_per_word)?;
    let groups_per_matrix = elements(OUTPUT, INPUT / GROUP)?;
    let packed = elements(EXPERTS, packed_per_matrix)?;
    let groups = elements(EXPERTS, groups_per_matrix)?;
    let mut input = pool.allocate_zeroed::<bf16>(stream, INPUT)?;
    let mut gate_weight = pool.allocate_zeroed::<u32>(stream, packed)?;
    let mut gate_scales = pool.allocate_zeroed::<bf16>(stream, groups)?;
    let mut gate_biases = pool.allocate_zeroed::<bf16>(stream, groups)?;
    let mut up_weight = pool.allocate_zeroed::<u32>(stream, packed)?;
    let mut up_scales = pool.allocate_zeroed::<bf16>(stream, groups)?;
    let mut up_biases = pool.allocate_zeroed::<bf16>(stream, groups)?;
    let gate = Bank {
        weight: &gate_weight,
        scales: &gate_scales,
        biases: &gate_biases,
    };
    let up = Bank {
        weight: &up_weight,
        scales: &up_scales,
        biases: &up_biases,
    };
    let mut selected_host = context.allocate_pinned::<u32>(SELECTED)?;
    selected_host.copy_from_slice(&[0, 1, 2, 3, 4, 5, 6, 7])?;
    let mut selected = pool.allocate::<u32>(stream, SELECTED)?;
    stream.copy_to_device(&mut selected_host, &mut selected)?;
    let mut single_gate = pool.allocate_zeroed::<bf16>(stream, OUTPUT)?;
    let mut single_up = pool.allocate_zeroed::<bf16>(stream, OUTPUT)?;
    let pair_output = elements(SELECTED, OUTPUT)?;
    let mut pair_gate = pool.allocate_zeroed::<bf16>(stream, pair_output)?;
    let mut pair_up = pool.allocate_zeroed::<bf16>(stream, pair_output)?;
    let compiler = Compiler::new(context.clone())?;
    let matrix = AffineGemvSpec::new(INPUT, OUTPUT, GROUP, BITS)?;
    let single = AffineQuantizedGemv::compile(&compiler, matrix)?;
    let pair = SelectedAffinePair::compile(
        &compiler,
        SelectedAffinePairSpec::new(matrix, EXPERTS, SELECTED)?,
    )?;

    for _ in 0..10 {
        execute_pair(
            &pair,
            stream,
            &input,
            &selected,
            [gate, up],
            &mut PairOutputs { gate: &mut pair_gate, up: &mut pair_up },
        )?;
    }
    stream.synchronize()?;
    let sequential_us = time(context, stream, || {
        for expert in 0..SELECTED {
            execute_single(&single, stream, &input, gate, &mut single_gate, expert)?;
            execute_single(&single, stream, &input, up, &mut single_up, expert)?;
        }
        Ok(())
    })?;
    let fused_us = time(context, stream, || {
        execute_pair(
            &pair,
            stream,
            &input,
            &selected,
            [gate, up],
            &mut PairOutputs { gate: &mut pair_gate, up: &mut pair_up },
        )
    })?;
    let bank_bytes =
        packed_per_matrix * size_of::<u32>() + groups_per_matrix * size_of::<bf16>() * 2;
    let operation_bytes = SELECTED * bank_bytes * 2 + INPUT * size_of::<bf16>();
    let bytes = u32::try_from(operation_bytes).map_or(f64::NAN, f64::from);
    std::hint::black_box((
        &mut input, &mut gate_weight, &mut gate_scales, &mut gate_biases, &mut up_weight,
        &mut up_scales, &mut up_biases,
    ));
    Ok(Report {
        sequential_us,
        fused_us,
        bandwidth: bytes / (fused_us * 1_000.0),
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

fn execute_single(
    operation: &AffineQuantizedGemv,
    stream: &Stream,
    input: &DeviceBuffer<bf16>,
    bank: Bank<'_>,
    output: &mut DeviceBuffer<bf16>,
    expert: usize,
) -> libmir_cuda::Result<()> {
    operation.execute(
        stream,
        &mut AffineGemvLaunch {
            input,
            weight: bank.weight,
            scales: bank.scales,
            biases: bank.biases,
            output,
            matrix_index: expert,
        },
    )
}

fn execute_pair(
    operation: &SelectedAffinePair,
    stream: &Stream,
    input: &DeviceBuffer<bf16>,
    selected: &DeviceBuffer<u32>,
    banks: [Bank<'_>; 2],
    outputs: &mut PairOutputs<'_>,
) -> libmir_cuda::Result<()> {
    let [gate, up] = banks;
    operation.execute(
        stream,
        &mut SelectedAffinePairLaunch {
            input,
            selected,
            gate_weight: gate.weight,
            gate_scales: gate.scales,
            gate_biases: gate.biases,
            up_weight: up.weight,
            up_scales: up.scales,
            up_biases: up.biases,
            gate_output: &mut *outputs.gate,
            up_output: &mut *outputs.up,
        },
    )
}

fn elements(left: usize, right: usize) -> libmir_cuda::Result<usize> {
    left.checked_mul(right)
        .ok_or(libmir_cuda::Error::InvalidQuantizedGemv("shape overflow"))
}
