use mircuda::{DeviceBuffer, f16};

use super::execute::buffer;
use crate::Result;

pub(super) struct Scratch {
    pub tokens: usize,
    pub first: DeviceBuffer<f16>,
    pub second: DeviceBuffer<f16>,
    pub qkv_raw: DeviceBuffer<f16>,
    pub qkv: DeviceBuffer<f16>,
    pub attention: DeviceBuffer<f16>,
    pub projection_raw: DeviceBuffer<f16>,
    pub projection: DeviceBuffer<f16>,
    pub up_gate: DeviceBuffer<f16>,
    pub activated: DeviceBuffer<f16>,
    pub down_raw: DeviceBuffer<f16>,
    pub down: DeviceBuffer<f16>,
    pub cls: DeviceBuffer<f16>,
    pub pooled_raw: DeviceBuffer<f16>,
    pub pooled: DeviceBuffer<f16>,
    pub score_raw: DeviceBuffer<f16>,
    pub score: DeviceBuffer<f16>,
}

impl Scratch {
    pub fn new(
        backend: &crate::CudaBackend,
        tokens: usize,
        hidden: usize,
        intermediate: usize,
    ) -> Result<Self> {
        let alloc = |elements| buffer(backend, elements);
        Ok(Self {
            tokens,
            first: alloc(tokens * hidden)?,
            second: alloc(tokens * hidden)?,
            qkv_raw: alloc(tokens * hidden * 3)?,
            qkv: alloc(tokens * hidden * 3)?,
            attention: alloc(tokens * hidden)?,
            projection_raw: alloc(tokens * hidden)?,
            projection: alloc(tokens * hidden)?,
            up_gate: alloc(tokens * intermediate * 2)?,
            activated: alloc(tokens * intermediate)?,
            down_raw: alloc(tokens * hidden)?,
            down: alloc(tokens * hidden)?,
            cls: alloc(hidden)?,
            pooled_raw: alloc(hidden)?,
            pooled: alloc(hidden)?,
            score_raw: alloc(1)?,
            score: alloc(1)?,
        })
    }
}
