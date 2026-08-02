use mircuda::{DeviceBuffer, bf16};

use super::ClampedRoutedConfig;
use crate::{CudaBackend, Error, Result};

pub(super) struct ClampedRoutedScratch {
    pub rope_inverse: DeviceBuffer<f32>,
    pub rope_sines: DeviceBuffer<f32>,
    pub rope_cosines: DeviceBuffer<f32>,
    pub rope_concentration: f32,
    pub normalized: DeviceBuffer<bf16>,
    pub packed_qkv: DeviceBuffer<bf16>,
    pub raw_query: DeviceBuffer<bf16>,
    pub raw_key: DeviceBuffer<bf16>,
    pub raw_value: DeviceBuffer<bf16>,
    pub query: DeviceBuffer<bf16>,
    pub key: DeviceBuffer<bf16>,
    pub value: DeviceBuffer<bf16>,
    pub attended: DeviceBuffer<bf16>,
    pub projected: DeviceBuffer<bf16>,
    pub biased: DeviceBuffer<bf16>,
    pub residual: DeviceBuffer<bf16>,
    pub router: DeviceBuffer<bf16>,
    pub router_biased: DeviceBuffer<bf16>,
    pub selected: DeviceBuffer<u32>,
    pub routing: DeviceBuffer<bf16>,
    pub activated: DeviceBuffer<bf16>,
    pub route_partial: Option<DeviceBuffer<f32>>,
    pub moe: DeviceBuffer<bf16>,
}

impl ClampedRoutedScratch {
    pub(super) fn new(
        backend: &CudaBackend,
        config: ClampedRoutedConfig,
        tokens: usize,
        route_parallel: bool,
    ) -> Result<Self> {
        let bf16 = |elements| backend.inner.pool.allocate::<bf16>(&backend.inner.stream, elements);
        let hidden = product(tokens, config.hidden)?;
        let query = product(tokens, product(config.query_heads, config.head_dim)?)?;
        let kv = product(tokens, product(config.kv_heads, config.head_dim)?)?;
        let packed = query
            .checked_add(2 * kv)
            .ok_or(Error::InvalidDecoderKernel("clamped-routed QKV scratch overflow"))?;
        let routes = product(tokens, config.top_k)?;
        let route_partial_elements = product(routes, config.hidden)?;
        let half_head = config.head_dim / 2;
        let rope_values = product(tokens, half_head)?;
        Ok(Self {
            rope_inverse: upload_rope_inverse(backend, config)?,
            rope_sines: backend.inner.pool.allocate(&backend.inner.stream, rope_values)?,
            rope_cosines: backend.inner.pool.allocate(&backend.inner.stream, rope_values)?,
            rope_concentration: config.factor.ln().mul_add(0.1, 1.0),
            normalized: bf16(hidden)?,
            packed_qkv: bf16(packed)?,
            raw_query: bf16(query)?,
            raw_key: bf16(kv)?,
            raw_value: bf16(kv)?,
            query: bf16(query)?,
            key: bf16(kv)?,
            value: bf16(kv)?,
            attended: bf16(query)?,
            projected: bf16(hidden)?,
            biased: bf16(hidden)?,
            residual: bf16(hidden)?,
            router: bf16(product(tokens, config.experts)?)?,
            router_biased: bf16(product(tokens, config.experts)?)?,
            selected: backend.inner.pool.allocate(&backend.inner.stream, routes)?,
            routing: bf16(routes)?,
            activated: bf16(product(routes, config.intermediate)?)?,
            route_partial: route_parallel
                .then(|| backend.inner.pool.allocate(&backend.inner.stream, route_partial_elements))
                .transpose()?,
            moe: bf16(hidden)?,
        })
    }
}

fn product(left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right)
        .ok_or(Error::InvalidDecoderKernel("clamped-routed scratch size overflow"))
}

fn upload_rope_inverse(
    backend: &CudaBackend,
    config: ClampedRoutedConfig,
) -> Result<DeviceBuffer<f32>> {
    let half = config.head_dim / 2;
    let half_float = f32::from(u16::try_from(half)?);
    let head_float = f32::from(u16::try_from(config.head_dim)?);
    let theta_log = config.theta.ln();
    let circle = 2.0 * std::f32::consts::PI;
    let low = half_float * (config.initial_context / (config.beta_fast * circle)).ln() / theta_log;
    let high = half_float * (config.initial_context / (config.beta_slow * circle)).ln() / theta_log;
    let inverse = (0..half)
        .map(|pair| -> Result<f32> {
            let pair = f32::from(u16::try_from(pair)?);
            let frequency = config.theta.powf(2.0 * pair / head_float);
            let ramp = ((pair - low) / (high - low)).clamp(0.0, 1.0);
            Ok((1.0 - ramp) / frequency + ramp / (config.factor * frequency))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut staging = backend.inner.context.allocate_pinned(half)?;
    staging.copy_from_slice(&inverse)?;
    let mut device = backend.inner.pool.allocate(&backend.inner.stream, half)?;
    backend.inner.stream.copy_to_device(&mut staging, &mut device)?;
    Ok(device)
}
