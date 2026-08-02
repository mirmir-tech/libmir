use std::path::PathBuf;

use mircuda::{Compiler, Context, DeviceBuffer, DeviceElement, Driver, MemoryPool, Stream, bf16};

use super::*;
use crate::{Result, kernels::AffineGemvSpec};

#[test]
fn selected_affine_roles_support_native_mlx_widths() -> Result<()> {
    let fixture = Fixture::new()?;
    for bits in [2, 3, 4, 5, 6, 8] {
        check_pair(&fixture, bits)?;
        check_pair_batch(&fixture, bits)?;
        check_gated(&fixture, bits)?;
        check_reduce(&fixture, bits)?;
    }
    Ok(())
}

fn check_pair_batch(fixture: &Fixture, bits: usize) -> Result<()> {
    let input = fixture.copy(&[bf16::ONE; 128])?;
    let selected = fixture.copy(&[1_u32, 0, 0, 1])?;
    let weight = fixture.copy(&weights(bits))?;
    let scales = fixture.copy(&[bf16::ONE; 4])?;
    let biases = fixture.copy(&[bf16::ZERO; 4])?;
    let mut gate_output = fixture.pool.allocate_zeroed::<bf16>(&fixture.stream, 8)?;
    let mut up_output = fixture.pool.allocate_zeroed::<bf16>(&fixture.stream, 8)?;
    let operation = SelectedAffinePair::compile(
        &fixture.compiler,
        SelectedAffinePairSpec::new_batch(matrix(bits)?, 2, 2, 2)?,
    )?;
    operation.execute(
        &fixture.stream,
        &mut SelectedAffinePairLaunch {
            input: &input,
            selected: &selected,
            gate_weight: &weight,
            gate_scales: &scales,
            gate_biases: &biases,
            up_weight: &weight,
            up_scales: &scales,
            up_biases: &biases,
            gate_output: &mut gate_output,
            up_output: &mut up_output,
        },
    )?;
    let expected = [128.0, 192.0, 64.0, 128.0, 64.0, 128.0, 128.0, 192.0].map(bf16::from_f32);
    assert_eq!(fixture.read(&gate_output)?, expected);
    assert_eq!(fixture.read(&up_output)?, expected);
    Ok(())
}

fn check_pair(fixture: &Fixture, bits: usize) -> Result<()> {
    let data = Data::new(fixture, bits, false)?;
    let mut gate_output = fixture.pool.allocate_zeroed::<bf16>(&fixture.stream, 4)?;
    let mut up_output = fixture.pool.allocate_zeroed::<bf16>(&fixture.stream, 4)?;
    let operation = SelectedAffinePair::compile(
        &fixture.compiler,
        SelectedAffinePairSpec::new(matrix(bits)?, 2, 2)?,
    )?;
    operation.execute(
        &fixture.stream,
        &mut SelectedAffinePairLaunch {
            input: &data.input,
            selected: &data.selected,
            gate_weight: &data.weight,
            gate_scales: &data.scales,
            gate_biases: &data.biases,
            up_weight: &data.weight,
            up_scales: &data.scales,
            up_biases: &data.biases,
            gate_output: &mut gate_output,
            up_output: &mut up_output,
        },
    )?;
    let expected = expected_selected();
    assert_eq!(fixture.read(&gate_output)?, expected);
    assert_eq!(fixture.read(&up_output)?, expected);
    Ok(())
}

fn check_gated(fixture: &Fixture, bits: usize) -> Result<()> {
    let data = Data::new(fixture, bits, false)?;
    let mut output = fixture.pool.allocate_zeroed::<bf16>(&fixture.stream, 4)?;
    let operation = SelectedAffineGated::compile(
        &fixture.compiler,
        SelectedAffineGatedSpec::new(matrix(bits)?, 2, 2, GatedActivation::Silu)?,
    )?;
    operation.execute(
        &fixture.stream,
        &mut SelectedAffineGatedLaunch {
            input: &data.input,
            selected: &data.selected,
            gate_weight: &data.weight,
            gate_scales: &data.scales,
            gate_biases: &data.biases,
            up_weight: &data.weight,
            up_scales: &data.scales,
            up_biases: &data.biases,
            output: &mut output,
        },
    )?;
    assert_eq!(fixture.read(&output)?, [16384.0, 36864.0, 4096.0, 16384.0].map(bf16::from_f32));
    Ok(())
}

fn check_reduce(fixture: &Fixture, bits: usize) -> Result<()> {
    let data = Data::new(fixture, bits, true)?;
    let routing = fixture.copy(&[bf16::from_f32(0.5); 2])?;
    let mut output = fixture.pool.allocate_zeroed::<bf16>(&fixture.stream, 2)?;
    let operation = SelectedAffineReduce::compile(
        &fixture.compiler,
        SelectedAffineReduceSpec::new(matrix(bits)?, 2, 2)?,
    )?;
    operation.execute(
        &fixture.stream,
        &mut SelectedAffineReduceLaunch {
            input: &data.input,
            selected: &data.selected,
            routing_weights: &routing,
            weight: &data.weight,
            scales: &data.scales,
            biases: &data.biases,
            output: &mut output,
        },
    )?;
    assert_eq!(fixture.read(&output)?, [96.0, 160.0].map(bf16::from_f32));
    Ok(())
}

fn matrix(bits: usize) -> Result<AffineGemvSpec> {
    AffineGemvSpec::new(64, 2, 64, bits)
}

fn expected_selected() -> [bf16; 4] {
    [128.0, 192.0, 64.0, 128.0].map(bf16::from_f32)
}

struct Data {
    input: DeviceBuffer<bf16>,
    selected: DeviceBuffer<u32>,
    weight: DeviceBuffer<u32>,
    scales: DeviceBuffer<bf16>,
    biases: DeviceBuffer<bf16>,
}

impl Data {
    fn new(fixture: &Fixture, bits: usize, selected_input: bool) -> Result<Self> {
        let input_len = if selected_input {
            128
        } else {
            64
        };
        Ok(Self {
            input: fixture.copy(&vec![bf16::ONE; input_len])?,
            selected: fixture.copy(&[1_u32, 0])?,
            weight: fixture.copy(&weights(bits))?,
            scales: fixture.copy(&[bf16::ONE; 4])?,
            biases: fixture.copy(&[bf16::ZERO; 4])?,
        })
    }
}

fn weights(bits: usize) -> Vec<u32> {
    let values = [1_u32, 2, 2, 3].into_iter().flat_map(|value| [value; 64]).collect::<Vec<_>>();
    let mut packed = vec![0_u32; values.len() * bits / 32];
    for (index, value) in values.into_iter().enumerate() {
        let bit = index * bits;
        packed[bit / 32] |= value << (bit % 32);
        if bit % 32 + bits > 32 {
            packed[bit / 32 + 1] |= value >> (32 - bit % 32);
        }
    }
    packed
}

struct Fixture {
    context: Context,
    stream: Stream,
    pool: MemoryPool,
    compiler: Compiler,
}

impl Fixture {
    fn new() -> Result<Self> {
        let driver = Driver::initialize()?;
        let device = driver.devices()?.into_iter().next().ok_or(mircuda::Error::InvalidLaunch)?;
        let context = driver.create_context(device)?;
        Ok(Self {
            stream: context.create_stream()?,
            pool: context.default_memory_pool()?,
            compiler: Compiler::with_include_paths(
                context.clone(),
                [PathBuf::from("/usr/local/cuda/include")],
            )?,
            context,
        })
    }

    fn copy<T: DeviceElement + Copy>(&self, values: &[T]) -> Result<DeviceBuffer<T>> {
        let mut host = self.context.allocate_pinned::<T>(values.len())?;
        host.copy_from_slice(values)?;
        let mut device = self.pool.allocate::<T>(&self.stream, values.len())?;
        self.stream.copy_to_device(&mut host, &mut device)?;
        Ok(device)
    }

    fn read<T: DeviceElement + Copy>(&self, values: &DeviceBuffer<T>) -> Result<Vec<T>> {
        let mut host = self.context.allocate_pinned::<T>(values.len())?;
        self.stream.copy_to_host(values, &mut host)?;
        Ok(host.to_vec()?)
    }
}
