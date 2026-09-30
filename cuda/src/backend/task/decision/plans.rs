use std::{
    collections::{HashMap, hash_map::Entry},
    sync::Mutex,
};

use mircuda::{CublasGemmOffsets, CublasGemmPlan, CublasGemmSpec, DeviceBuffer, bf16};

use crate::{CudaBackend, Error, Result, kernels::DecisionElement};

type Cache<I> = Mutex<HashMap<CublasGemmSpec, CublasGemmPlan<I, f32>>>;

/// cuBLAS plans reused across calls, one per geometry and input type;
/// creating one costs an allocation and a device synchronisation.
#[derive(Default)]
pub struct Plans {
    f32: Cache<f32>,
    bf16: Cache<bf16>,
}

/// An input element type the decision products accept; outputs stay f32.
pub trait Input: DecisionElement {
    fn cache(plans: &Plans) -> &Cache<Self>;

    fn from_f32(value: f32) -> Self;
}

impl Input for f32 {
    fn cache(plans: &Plans) -> &Cache<Self> {
        &plans.f32
    }

    fn from_f32(value: f32) -> Self {
        value
    }
}

impl Input for bf16 {
    fn cache(plans: &Plans) -> &Cache<Self> {
        &plans.bf16
    }

    fn from_f32(value: f32) -> Self {
        Self::from_f32(value)
    }
}

/// Operands of one product and the element offset of each.
pub struct Operands<'a, I: Input> {
    pub left: &'a DeviceBuffer<I>,
    pub right: &'a DeviceBuffer<I>,
    pub output: &'a mut DeviceBuffer<f32>,
    pub offsets: CublasGemmOffsets,
}

impl Plans {
    /// Enqueues `output = alpha · op(left) · op(right) + beta · output`.
    pub fn multiply<I: Input>(
        &self,
        backend: &CudaBackend,
        spec: CublasGemmSpec,
        operands: Operands<'_, I>,
        scaling: (f32, f32),
    ) -> Result<()> {
        let mut plans = I::cache(self)
            .lock()
            .map_err(|_| Error::InvalidDecoderKernel("decision plan cache is poisoned"))?;
        let stream = &backend.inner.stream;
        let plan = match plans.entry(spec) {
            Entry::Occupied(plan) => plan.into_mut(),
            Entry::Vacant(slot) => {
                slot.insert(CublasGemmPlan::new(&backend.inner.context, stream, spec)?)
            },
        };
        let Operands { left, right, output, offsets } = operands;
        let executed = plan.execute(stream, (left, right, output), offsets, scaling);
        drop(plans);
        Ok(executed?)
    }
}
