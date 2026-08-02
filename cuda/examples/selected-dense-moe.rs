use std::io::{self, Write};

use libmir_cuda::kernels::{
    DenseGateUpLayout, DenseGatedActivation, SelectedDenseGateLaunch, SelectedDenseMoe,
    SelectedDenseMoeSpec, SelectedDenseReduceLaunch,
};
use mircuda::{Compiler, Context, DeviceBuffer, Driver, MemoryPool, Stream, bf16};

const HIDDEN: usize = 2_880;
const INTERMEDIATE: usize = 2_880;
const EXPERTS: usize = 32;
const SELECTED: usize = 4;
const ITERATIONS: u16 = 100;

struct Buffers<'a> {
    input: &'a DeviceBuffer<bf16>,
    selected: &'a DeviceBuffer<u32>,
    routing: &'a DeviceBuffer<bf16>,
    gate_up: &'a DeviceBuffer<bf16>,
    gate_up_bias: &'a DeviceBuffer<bf16>,
    down: &'a DeviceBuffer<bf16>,
    down_bias: &'a DeviceBuffer<bf16>,
    intermediate: &'a mut DeviceBuffer<bf16>,
    partial: &'a mut DeviceBuffer<f32>,
    output: &'a mut DeviceBuffer<bf16>,
}

struct Report {
    gated_us: f64,
    reduce_us: f64,
    full_us: f64,
    gated_bandwidth: f64,
    reduce_bandwidth: f64,
    full_bandwidth: f64,
}

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
    writeln!(output, "GPT-OSS expert MLP: top-k={SELECTED}, BF16, fused interleaved gate/up")?;
    writeln!(
        output,
        "  selected gate/up + activation: {:.3} us/token ({:.3} GB/s)",
        report.gated_us, report.gated_bandwidth
    )?;
    writeln!(
        output,
        "  selected down + reduction:     {:.3} us/token ({:.3} GB/s)",
        report.reduce_us, report.reduce_bandwidth
    )?;
    writeln!(
        output,
        "  complete expert path:          {:.3} us/token ({:.3} GB/s)",
        report.full_us, report.full_bandwidth
    )?;
    Ok(())
}

fn profile(context: &Context, stream: &Stream, pool: &MemoryPool) -> libmir_cuda::Result<Report> {
    let input = pool.allocate_zeroed::<bf16>(stream, HIDDEN)?;
    let selected = upload_selected(context, stream, pool)?;
    let routing = upload_routing(context, stream, pool)?;
    let gate_up_elements = EXPERTS * INTERMEDIATE * 2 * HIDDEN;
    let gate_up = pool.allocate_zeroed::<bf16>(stream, gate_up_elements)?;
    let gate_up_bias = pool.allocate_zeroed::<bf16>(stream, EXPERTS * INTERMEDIATE * 2)?;
    let down = pool.allocate_zeroed::<bf16>(stream, EXPERTS * HIDDEN * INTERMEDIATE)?;
    let down_bias = pool.allocate_zeroed::<bf16>(stream, EXPERTS * HIDDEN)?;
    let mut intermediate = pool.allocate_zeroed::<bf16>(stream, SELECTED * INTERMEDIATE)?;
    let mut partial = pool.allocate_zeroed::<f32>(stream, SELECTED * HIDDEN)?;
    let mut output = pool.allocate_zeroed::<bf16>(stream, HIDDEN)?;
    let operation = SelectedDenseMoe::compile(
        &Compiler::new(context.clone())?,
        SelectedDenseMoeSpec {
            tokens: 1,
            input_features: HIDDEN,
            output_features: INTERMEDIATE,
            expert_count: EXPERTS,
            selected_count: SELECTED,
            gate_up_layout: DenseGateUpLayout::FusedInterleaved,
            gate_transposed: true,
            up_transposed: true,
            down_transposed: true,
            gate_bias: true,
            up_bias: true,
            down_bias: true,
            activation: DenseGatedActivation::clamped_silu(1.702, 7.0, 1.0),
        },
    )?;
    let mut buffers = Buffers {
        input: &input,
        selected: &selected,
        routing: &routing,
        gate_up: &gate_up,
        gate_up_bias: &gate_up_bias,
        down: &down,
        down_bias: &down_bias,
        intermediate: &mut intermediate,
        partial: &mut partial,
        output: &mut output,
    };
    for _ in 0..10 {
        execute_full(&operation, stream, &mut buffers)?;
    }
    stream.synchronize()?;
    let gated_us = time(context, stream, || execute_gated(&operation, stream, &mut buffers))?;
    let reduce_us = time(context, stream, || execute_reduce(&operation, stream, &mut buffers))?;
    let full_us = time(context, stream, || execute_full(&operation, stream, &mut buffers))?;
    let gated_bytes = SELECTED * 2 * HIDDEN * INTERMEDIATE * size_of::<bf16>();
    let reduce_bytes = SELECTED * HIDDEN * INTERMEDIATE * size_of::<bf16>();
    let gated_rate = bandwidth(gated_bytes, gated_us)?;
    let reduce_rate = bandwidth(reduce_bytes, reduce_us)?;
    Ok(Report {
        gated_us,
        reduce_us,
        full_us,
        gated_bandwidth: gated_rate,
        reduce_bandwidth: reduce_rate,
        full_bandwidth: bandwidth(gated_bytes + reduce_bytes, full_us)?,
    })
}

fn upload_selected(
    context: &Context,
    stream: &Stream,
    pool: &MemoryPool,
) -> mircuda::Result<DeviceBuffer<u32>> {
    let mut host = context.allocate_pinned::<u32>(SELECTED)?;
    host.copy_from_slice(&[0, 1, 2, 3])?;
    let mut device = pool.allocate::<u32>(stream, SELECTED)?;
    stream.copy_to_device(&mut host, &mut device)?;
    Ok(device)
}

fn upload_routing(
    context: &Context,
    stream: &Stream,
    pool: &MemoryPool,
) -> mircuda::Result<DeviceBuffer<bf16>> {
    let mut host = context.allocate_pinned::<bf16>(SELECTED)?;
    host.copy_from_slice(&[bf16::from_f32(0.25); SELECTED])?;
    let mut device = pool.allocate::<bf16>(stream, SELECTED)?;
    stream.copy_to_device(&mut host, &mut device)?;
    Ok(device)
}

fn execute_full(
    operation: &SelectedDenseMoe,
    stream: &Stream,
    buffers: &mut Buffers<'_>,
) -> libmir_cuda::Result<()> {
    execute_gated(operation, stream, buffers)?;
    execute_reduce(operation, stream, buffers)
}

fn execute_gated(
    operation: &SelectedDenseMoe,
    stream: &Stream,
    buffers: &mut Buffers<'_>,
) -> libmir_cuda::Result<()> {
    operation.gated(
        stream,
        &mut SelectedDenseGateLaunch {
            input: buffers.input,
            selected: buffers.selected,
            gate_weight: buffers.gate_up,
            gate_bias: buffers.gate_up_bias,
            up_weight: buffers.gate_up,
            up_bias: buffers.gate_up_bias,
            output: &mut *buffers.intermediate,
        },
    )
}

fn execute_reduce(
    operation: &SelectedDenseMoe,
    stream: &Stream,
    buffers: &mut Buffers<'_>,
) -> libmir_cuda::Result<()> {
    operation.reduce(
        stream,
        &mut SelectedDenseReduceLaunch {
            input: &*buffers.intermediate,
            selected: buffers.selected,
            routing: buffers.routing,
            weight: buffers.down,
            bias: buffers.down_bias,
            partial: &mut *buffers.partial,
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

fn bandwidth(bytes: usize, microseconds: f64) -> libmir_cuda::Result<f64> {
    let Ok(bytes) = u32::try_from(bytes) else {
        return Err(libmir_cuda::Error::InvalidDecoderKernel(
            "profiled expert byte count exceeds u32",
        ));
    };
    Ok(f64::from(bytes) / (microseconds * 1_000.0))
}
