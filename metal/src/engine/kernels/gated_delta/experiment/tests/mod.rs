use std::io::Write;

use super::*;

mod decode;
mod reference;

#[derive(Clone, Copy, Debug)]
struct Case {
    batch: usize,
    steps: usize,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    dtype: DType,
}

impl Case {
    fn inputs(self, stream: &Stream) -> Result<[Array; 6]> {
        let Self {
            batch: b,
            steps: t,
            key_heads: hk,
            value_heads: hv,
            key_dim: dk,
            value_dim: dv,
            dtype,
        } = self;
        Ok([
            data(stream, [b, t, hk, dk], dtype, 1, 0.08)?,
            data(stream, [b, t, hk, dk], dtype, 2, 0.08)?,
            data(stream, [b, t, hv, dv], dtype, 3, 0.5)?,
            stream.graph().sigmoid(&data(stream, [b, t, hv], DType::Float32, 4, 0.3)?)?,
            stream.graph().sigmoid(&data(stream, [b, t, hv], DType::Float32, 5, 0.6)?)?,
            data(stream, [b, hv, dv, dk], DType::Float32, 6, 0.3)?,
        ])
    }
}

fn data<const N: usize>(
    stream: &Stream,
    shape: [usize; N],
    dtype: DType,
    mut seed: u32,
    scale: f32,
) -> Result<Array> {
    let values = (0..shape.iter().product())
        .map(|_| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            Ok((f32::from(u16::try_from(seed >> 16)?) / 32768.0 - 1.0) * scale)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(stream.graph().astype(&Array::from_slice(&values, shape)?, dtype)?)
}

fn read(stream: &Stream, value: &Array) -> Result<Vec<f32>> {
    Ok(stream.read(&stream.graph().astype(value, DType::Float32)?)?)
}

fn exact(stream: &Stream, a: &[Array; 2], b: &[Array; 2], label: &str) -> Result<()> {
    for (index, (a, b)) in a.iter().zip(b).enumerate() {
        let a = read(stream, a)?;
        let b = read(stream, b)?;
        let mismatches = a.iter().zip(&b).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        assert_eq!(mismatches, 0, "{label}, output/state {index}, length {}", a.len());
    }
    Ok(())
}

#[test]
fn packed_recurrence_matches_explicit_tree_and_native() -> Result<()> {
    let stream = mirtal::Device::gpu(0).new_stream()?;
    let kernels = Recurrence::new()?;
    for (b, hk, hv, dv, dtype) in [
        (1, 16, 32, 128, DType::Bfloat16),
        (3, 2, 8, 64, DType::Bfloat16),
        (5, 2, 4, 128, DType::Bfloat16),
        (1, 2, 2, 256, DType::Float16),
        (1, 2, 4, 128, DType::Float32),
        (1, 1, 2, 8, DType::Bfloat16),
        (1, 1, 2, 24, DType::Bfloat16),
    ] {
        for steps in [1, 7, 64, 257] {
            let case = Case {
                batch: b,
                steps,
                key_heads: hk,
                value_heads: hv,
                key_dim: 128,
                value_dim: dv,
                dtype,
            };
            let inputs = case.inputs(&stream)?;
            let native = kernels.run(&stream, inputs.each_ref(), Layout::Native)?;
            let explicit = kernels.run(&stream, inputs.each_ref(), Layout::Explicit)?;
            let packed = kernels.run(&stream, inputs.each_ref(), Layout::Packed)?;
            exact(&stream, &explicit, &packed, "explicit versus packed")?;
            exact(&stream, &native, &explicit, "native reduction canary")?;
            reference::check(&stream, case, &inputs, &packed)?;
            writeln!(std::io::stderr().lock(), "gdn.numerical: {case:?} exact output/state")?;
        }
    }
    Ok(())
}

#[test]
fn packed_recurrence_preserves_continuation_and_snapshot() -> Result<()> {
    let stream = mirtal::Device::gpu(0).new_stream()?;
    let kernels = Recurrence::new()?;
    let case = Case {
        batch: 3,
        steps: 7,
        key_heads: 2,
        value_heads: 4,
        key_dim: 128,
        value_dim: 128,
        dtype: DType::Bfloat16,
    };
    let mut inputs = case.inputs(&stream)?;
    let saved = inputs[5].clone();
    let saved_values = read(&stream, &saved)?;
    let mut native = kernels.run(&stream, inputs.each_ref(), Layout::Native)?;
    let mut packed = kernels.run(&stream, inputs.each_ref(), Layout::Packed)?;
    exact(&stream, &native, &packed, "initialized prefill")?;
    inputs = Case { steps: 1, ..case }.inputs(&stream)?;
    for step in 0..128 {
        inputs[5] = native[1].clone();
        native = kernels.run(&stream, inputs.each_ref(), Layout::Native)?;
        inputs[5] = packed[1].clone();
        packed = kernels.run(&stream, inputs.each_ref(), Layout::Packed)?;
        exact(&stream, &native, &packed, &format!("continuation {step}"))?;
        stream.synchronize()?;
        for array in native.iter().chain(&packed) {
            stream.detach_graph(array)?;
        }
    }
    assert_eq!(saved_values, read(&stream, &saved)?);
    inputs[5] = saved;
    let restored = kernels.run(&stream, inputs.each_ref(), Layout::Packed)?;
    let expected = kernels.run(&stream, inputs.each_ref(), Layout::Native)?;
    exact(&stream, &restored, &expected, "restored checkpoint")
}

#[test]
fn packed_recurrence_falls_back_for_unsupported_dimensions() -> Result<()> {
    let stream = mirtal::Device::gpu(0).new_stream()?;
    let kernels = Recurrence::new()?;
    for (dk, dv) in [(64, 64), (128, 7)] {
        let inputs = Case {
            batch: 1,
            steps: 7,
            key_heads: 1,
            value_heads: 2,
            key_dim: dk,
            value_dim: dv,
            dtype: DType::Float32,
        }
        .inputs(&stream)?;
        let native = kernels.run(&stream, inputs.each_ref(), Layout::Native)?;
        let packed = kernels.run(&stream, inputs.each_ref(), Layout::Packed)?;
        exact(&stream, &native, &packed, "fallback")?;
    }
    Ok(())
}
