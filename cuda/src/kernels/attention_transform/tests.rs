use super::*;
use crate::{
    CudaBackend, CudaConfig,
    kernels::{GatedAttentionSplit, Mrope, ShiftedRmsNorm},
};

#[test]
fn fused_attention_transform_is_bitwise_equal_to_materialized_bf16_stages() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    for (tokens, head_dim, rotary_dim) in [(1, 32, 16), (5, 128, 64), (17, 256, 64), (3, 320, 192)]
    {
        for interleaved in [false, true] {
            for shift in [0.0, 1.0] {
                compare(&backend, tokens, head_dim, rotary_dim, interleaved, shift)?;
            }
        }
    }
    Ok(())
}

fn compare(
    backend: &CudaBackend,
    tokens: usize,
    head_dim: usize,
    rotary_dim: usize,
    interleaved: bool,
    shift: f32,
) -> Result<()> {
    let stream = backend.stream();
    let compiler = backend.compiler();
    let query_heads = 3;
    let key_heads = 2;
    let half = rotary_dim / 2;
    let spec = MropeSpec {
        tokens,
        heads: query_heads,
        head_dim,
        rotary_dim,
        sections: [half / 2, half / 4, half - half / 2 - half / 4],
        interleaved,
        theta: 10_000.0,
    };
    let allocate = |elements| backend.pool().allocate::<bf16>(stream, elements);
    let query = pattern(backend, tokens * query_heads * head_dim * 2, 1)?;
    let key = pattern(backend, tokens * key_heads * head_dim, 3)?;
    let qw = pattern(backend, head_dim, 5)?;
    let kw = pattern(backend, head_dim, 7)?;
    let positions = (0..3 * tokens)
        .map(|i| u32::try_from(i * 31 + 8192))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let mut host = backend.context().allocate_pinned(positions.len())?;
    host.copy_from_slice(&positions)?;
    let mut device_positions = backend.pool().allocate(stream, positions.len())?;
    stream.copy_to_device(&mut host, &mut device_positions)?;
    let mut q = allocate(tokens * query_heads * head_dim)?;
    let mut g = allocate(q.len())?;
    let mut normalized_query = allocate(q.len())?;
    let mut normalized_key = allocate(key.len())?;
    let mut expected_q = allocate(q.len())?;
    let mut expected_k = allocate(key.len())?;
    GatedAttentionSplit::compile(compiler, tokens, query_heads, head_dim)?
        .execute(stream, &query, &mut q, &mut g)?;
    ShiftedRmsNorm::compile(compiler, tokens * query_heads, head_dim, 1e-6, shift)?.execute(
        stream,
        &q,
        &qw,
        &mut normalized_query,
    )?;
    ShiftedRmsNorm::compile(compiler, tokens * key_heads, head_dim, 1e-6, shift)?.execute(
        stream,
        &key,
        &kw,
        &mut normalized_key,
    )?;
    Mrope::compile(compiler, spec)?.execute(
        stream,
        &normalized_query,
        &device_positions,
        &mut expected_q,
    )?;
    Mrope::compile(compiler, MropeSpec { heads: key_heads, ..spec })?.execute(
        stream,
        &normalized_key,
        &device_positions,
        &mut expected_k,
    )?;
    let mut actual_q = allocate(q.len())?;
    let mut actual_k = allocate(key.len())?;
    let mut actual_g = allocate(q.len())?;
    AttentionTransform::compile(compiler, spec, key_heads, 1e-6, shift)?.execute(
        stream,
        &query,
        &key,
        &qw,
        &kw,
        &device_positions,
        &mut actual_q,
        &mut actual_k,
        &mut actual_g,
    )?;
    for (expected, actual) in [(&expected_q, &actual_q), (&expected_k, &actual_k), (&g, &actual_g)]
    {
        assert_eq!(
            read(backend, expected)?,
            read(backend, actual)?,
            "tokens={tokens} dim={head_dim} interleaved={interleaved} shift={shift}"
        );
    }
    Ok(())
}

fn pattern(backend: &CudaBackend, elements: usize, seed: usize) -> Result<DeviceBuffer<bf16>> {
    let values = (0..elements)
        .map(|i| {
            let signed = i16::try_from((i * 17 + seed) % 127).unwrap_or_default() - 63;
            bf16::from_f32(f32::from(signed) / 64.0)
        })
        .collect::<Vec<_>>();
    let mut host = backend.context().allocate_pinned(elements)?;
    host.copy_from_slice(&values)?;
    let mut device = backend.pool().allocate(backend.stream(), elements)?;
    backend.stream().copy_to_device(&mut host, &mut device)?;
    Ok(device)
}

fn read(backend: &CudaBackend, device: &DeviceBuffer<bf16>) -> Result<Vec<bf16>> {
    let mut host = backend.context().allocate_pinned(device.len())?;
    backend.stream().copy_to_host(device, &mut host)?;
    Ok(host.to_vec()?)
}
