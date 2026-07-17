use mircuda::{DeviceBuffer, KernelNode, LaunchConfig, Stream, bf16};

use self::runtime::Runtime;
use super::*;

mod nvfp4;
mod runtime;

#[test]
fn rms_norm_matches_cpu_reference() -> Result<()> {
    let runtime = Runtime::new()?;
    let values = (0..128)
        .map(|index| Ok(bf16::from_f32(f32::from(u16::try_from(index % 13)?) - 6.0)))
        .collect::<Result<Vec<_>>>()?;
    let weights = vec![bf16::from_f32(0.75); 64];
    let input = runtime.copy(&values)?;
    let weight = runtime.copy(&weights)?;
    let mut output = runtime.pool.allocate::<bf16>(&runtime.stream, values.len())?;
    RmsNorm::compile(&runtime.compiler, 2, 64, 1.0e-6)?
        .execute(&runtime.stream, &input, &weight, &mut output)?;
    let actual = runtime.read(&output)?;
    for (row, chunk) in values.as_chunks::<64>().0.iter().enumerate() {
        let sum = chunk
            .iter()
            .fold(0.0_f32, |sum, value| value.to_f32().mul_add(value.to_f32(), sum));
        let inverse = (sum / 64.0 + 1.0e-6).sqrt().recip();
        for (column, value) in chunk.iter().enumerate() {
            let expected = bf16::from_f32(value.to_f32() * inverse * 0.75);
            assert_eq!(actual[row * 64 + column], expected);
        }
    }
    Ok(())
}

#[test]
fn partial_rope_matches_cpu_reference() -> Result<()> {
    let runtime = Runtime::new()?;
    let spec = RopeSpec {
        tokens: 2,
        heads: 1,
        head_dim: 8,
        rotary_dim: 4,
        pairing_dim: 4,
        theta: 10_000.0,
    };
    let values = (0..16)
        .map(|index| Ok(bf16::from_f32(f32::from(u16::try_from(index)?) / 8.0 - 1.0)))
        .collect::<Result<Vec<_>>>()?;
    let input = runtime.copy(&values)?;
    let mut output = runtime.pool.allocate::<bf16>(&runtime.stream, values.len())?;
    Rope::compile(&runtime.compiler, spec)?.execute(&runtime.stream, &input, &mut output, 3)?;
    let actual = runtime.read(&output)?;
    for token in 0..2 {
        for dimension in 0..8 {
            let index = token * 8 + dimension;
            let expected = rope_value(&values, token, dimension, spec, 3)?;
            assert!((actual[index].to_f32() - expected).abs() < 0.02);
        }
    }
    Ok(())
}

#[test]
fn proportional_rope_uses_full_head_pairing() -> Result<()> {
    let runtime = Runtime::new()?;
    let spec = RopeSpec {
        tokens: 1,
        heads: 1,
        head_dim: 8,
        rotary_dim: 4,
        pairing_dim: 8,
        theta: 10_000.0,
    };
    let values = (0..8)
        .map(|index| Ok(bf16::from_f32(f32::from(u16::try_from(index)?) / 4.0 - 1.0)))
        .collect::<Result<Vec<_>>>()?;
    let input = runtime.copy(&values)?;
    let mut output = runtime.pool.allocate::<bf16>(&runtime.stream, values.len())?;
    Rope::compile(&runtime.compiler, spec)?.execute(&runtime.stream, &input, &mut output, 3)?;
    let actual = runtime.read(&output)?;
    for (dimension, actual) in actual.iter().enumerate() {
        let expected = rope_value(&values, 0, dimension, spec, 3)?;
        assert!((actual.to_f32() - expected).abs() < 0.02);
    }
    Ok(())
}

#[test]
fn rope_graph_rebinds_position_without_recapture() -> Result<()> {
    let runtime = Runtime::new()?;
    let spec = RopeSpec {
        tokens: 1,
        heads: 1,
        head_dim: 8,
        rotary_dim: 4,
        pairing_dim: 4,
        theta: 10_000.0,
    };
    let values = (0..8)
        .map(|index| Ok(bf16::from_f32(f32::from(u16::try_from(index)?) / 4.0 - 1.0)))
        .collect::<Result<Vec<_>>>()?;
    let input = runtime.copy(&values)?;
    let mut output = runtime.pool.allocate::<bf16>(&runtime.stream, values.len())?;
    let rope = Rope::compile(&runtime.compiler, spec)?;
    let kernel = rope.kernel.clone();
    let resources = RopeGraphResources {
        rope,
        stream: &runtime.stream,
        input: &input,
        output: &mut output,
        position: 3,
        next_position: 7,
        tokens: narrow(spec.tokens)?,
        heads: narrow(spec.heads)?,
        head_dim: narrow(spec.head_dim)?,
        rotary_dim: narrow(spec.rotary_dim)?,
        pairing_dim: narrow(spec.pairing_dim)?,
        node: None,
    };
    {
        let mut graph = runtime.stream.capture(resources, capture_rope)?;
        let node =
            graph.resources().node.ok_or(Error::InvalidDecoderKernel("missing RoPE node"))?;
        graph.update_kernel(&node, &kernel, rope_config(spec)?, rebind_rope)?;
        graph.launch(&runtime.stream)?;
    }
    let actual = runtime.read(&output)?;
    for (dimension, actual) in actual.iter().enumerate().take(8) {
        let expected = rope_value(&values, 0, dimension, spec, 7)?;
        assert!((actual.to_f32() - expected).abs() < 0.02);
    }
    Ok(())
}

type RopeArguments<'a> = (
    &'a DeviceBuffer<bf16>,
    &'a mut DeviceBuffer<bf16>,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    f32,
);

struct RopeGraphResources<'a> {
    rope: Rope,
    stream: &'a Stream,
    input: &'a DeviceBuffer<bf16>,
    output: &'a mut DeviceBuffer<bf16>,
    position: u32,
    next_position: u32,
    tokens: u32,
    heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    pairing_dim: u32,
    node: Option<KernelNode<RopeKernel>>,
}

fn capture_rope(resources: &mut RopeGraphResources<'_>) -> Result<()> {
    resources.node = Some(resources.rope.execute_captured(
        resources.stream,
        resources.input,
        resources.output,
        usize::try_from(resources.position)?,
    )?);
    Ok(())
}

fn rebind_rope<'borrow>(resources: &'borrow mut RopeGraphResources<'_>) -> RopeArguments<'borrow> {
    resources.position = resources.next_position;
    (
        resources.input,
        resources.output,
        resources.tokens,
        resources.heads,
        resources.head_dim,
        resources.rotary_dim,
        resources.pairing_dim,
        resources.position,
        resources.rope.spec.theta,
    )
}

fn rope_config(spec: RopeSpec) -> Result<LaunchConfig> {
    let elements = product(product(spec.tokens, spec.heads)?, spec.head_dim)?;
    Ok(LaunchConfig {
        grid: (narrow(elements.div_ceil(256))?, 1, 1),
        block: (256, 1, 1),
        shared_memory_bytes: 0,
    })
}

fn rope_value(
    values: &[bf16],
    token: usize,
    dimension: usize,
    spec: RopeSpec,
    start: usize,
) -> Result<f32> {
    let half = spec.pairing_dim / 2;
    let pair = dimension % half;
    if dimension >= spec.pairing_dim || pair >= spec.rotary_dim / 2 {
        return Ok(values[token * spec.head_dim + dimension].to_f32());
    }
    let offset = token * spec.head_dim;
    let first = values[offset + pair].to_f32();
    let second = values[offset + pair + half].to_f32();
    let pair = f32::from(u16::try_from(pair)?);
    let pairing = f32::from(u16::try_from(spec.pairing_dim)?);
    let position = f32::from(u16::try_from(start + token)?);
    let angle = position * spec.theta.powf(-2.0 * pair / pairing);
    Ok(if dimension < half {
        (-second).mul_add(angle.sin(), first * angle.cos())
    } else {
        first.mul_add(angle.sin(), second * angle.cos())
    })
}
