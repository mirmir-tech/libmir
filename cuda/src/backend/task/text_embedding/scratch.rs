use mircuda::{DeviceBuffer, bf16};

use super::execute::buffer;
use crate::{
    Result,
    kernels::{ElementwiseBf16, PackedGatedBf16},
};

pub(super) struct Scratch {
    pub tokens: usize,
    pub query_width: usize,
    pub kv_width: usize,
    pub normalized: DeviceBuffer<bf16>,
    pub query: DeviceBuffer<bf16>,
    pub key: DeviceBuffer<bf16>,
    pub value: DeviceBuffer<bf16>,
    pub query_norm: DeviceBuffer<bf16>,
    pub key_norm: DeviceBuffer<bf16>,
    pub query_rope: DeviceBuffer<bf16>,
    pub key_rope: DeviceBuffer<bf16>,
    pub attention: DeviceBuffer<bf16>,
    pub projection: DeviceBuffer<bf16>,
    pub residual: DeviceBuffer<bf16>,
    pub gate: DeviceBuffer<bf16>,
    pub up: DeviceBuffer<bf16>,
    pub activated: DeviceBuffer<bf16>,
    pub mlp: DeviceBuffer<bf16>,
    pub selected: DeviceBuffer<bf16>,
    pub hidden_ops: ElementwiseBf16,
    pub gated: PackedGatedBf16,
}

impl Scratch {
    pub fn new(
        backend: &crate::CudaBackend,
        tokens: usize,
        hidden: usize,
        intermediate: usize,
        query: usize,
        kv: usize,
    ) -> Result<Self> {
        let alloc = |elements| buffer(backend, elements);
        Ok(Self {
            tokens,
            query_width: query,
            kv_width: kv,
            normalized: alloc(tokens * hidden)?,
            query: alloc(tokens * query)?,
            key: alloc(tokens * kv)?,
            value: alloc(tokens * kv)?,
            query_norm: alloc(tokens * query)?,
            key_norm: alloc(tokens * kv)?,
            query_rope: alloc(tokens * query)?,
            key_rope: alloc(tokens * kv)?,
            attention: alloc(tokens * query)?,
            projection: alloc(tokens * hidden)?,
            residual: alloc(tokens * hidden)?,
            gate: alloc(tokens * intermediate)?,
            up: alloc(tokens * intermediate)?,
            activated: alloc(tokens * intermediate)?,
            mlp: alloc(tokens * hidden)?,
            selected: alloc(hidden)?,
            hidden_ops: ElementwiseBf16::compile(&backend.inner.compiler, tokens * hidden)?,
            gated: PackedGatedBf16::compile(&backend.inner.compiler, tokens, intermediate)?,
        })
    }
}
