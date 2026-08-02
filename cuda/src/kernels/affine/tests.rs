use std::path::PathBuf;

use mircuda::{Compiler, Context, DeviceBuffer, DeviceElement, Driver, MemoryPool, Stream, bf16};

use super::*;

#[test]
fn gemv_and_embedding_support_native_mlx_widths() -> Result<()> {
    for bits in [2, 3, 4, 5, 6, 8] {
        check_gemv(bits)?;
        check_embedding(bits)?;
    }
    Ok(())
}

fn check_gemv(bits: usize) -> Result<()> {
    let fixture = Fixture::new()?;
    let input = fixture.copy(&[bf16::ONE; 64])?;
    let weight = fixture.copy(&weights(bits))?;
    let scales = fixture.copy(&[bf16::ONE; 2])?;
    let biases = fixture.copy(&[bf16::ZERO; 2])?;
    let mut output = fixture.pool.allocate_zeroed::<bf16>(&fixture.stream, 2)?;
    let operation =
        AffineQuantizedGemv::compile(&fixture.compiler, AffineGemvSpec::new(64, 2, 64, bits)?)?;
    operation.execute(
        &fixture.stream,
        &mut AffineGemvLaunch {
            input: &input,
            weight: &weight,
            scales: &scales,
            biases: &biases,
            output: &mut output,
            matrix_index: 0,
        },
    )?;
    assert_eq!(fixture.read(&output)?, [bf16::from_f32(64.0), bf16::from_f32(128.0)]);
    Ok(())
}

fn check_embedding(bits: usize) -> Result<()> {
    let fixture = Fixture::new()?;
    let weight = fixture.copy(&weights(bits))?;
    let scales = fixture.copy(&[bf16::ONE; 2])?;
    let biases = fixture.copy(&[bf16::ZERO; 2])?;
    let selected = fixture.copy(&[1_u32, 0])?;
    let mut output = fixture.pool.allocate_zeroed::<bf16>(&fixture.stream, 128)?;
    let operation = AffineEmbedding::compile(
        &fixture.compiler,
        AffineEmbeddingSpec {
            vocab: 2,
            hidden: 64,
            group_size: 64,
            bits,
            output_scale: 1.0,
        },
    )?;
    operation.execute(&fixture.stream, &weight, &scales, &biases, &selected, 0, 2, &mut output)?;
    let actual = fixture.read(&output)?;
    assert_eq!(actual[..64], [bf16::from_f32(2.0); 64]);
    assert_eq!(actual[64..], [bf16::ONE; 64]);
    Ok(())
}

fn weights(bits: usize) -> Vec<u32> {
    let values = [vec![1_u32; 64], vec![2_u32; 64]].concat();
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
