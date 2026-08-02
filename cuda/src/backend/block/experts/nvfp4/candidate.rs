use mircuda::{DeviceBuffer, bf16};

use crate::{
    BucketedNvFp4MoeBf16, CudaBackend, DirectNvFp4MoeBf16, GatedActivation, GroupedNvFp4MoeBf16,
    HybridNvFp4MoeBf16, MoeExecution, MoePlanRequest, NvFp4ExpertBank, Result,
    SelectedNvFp4MoeBf16,
};

#[derive(Debug)]
pub(super) struct Candidate {
    pub(super) execution: MoeExecution,
    pub(super) plan: Plan,
}

#[derive(Debug)]
pub(super) enum Plan {
    Bucketed(Box<BucketedNvFp4MoeBf16>),
    Direct(Box<DirectNvFp4MoeBf16>),
    Grouped(Box<GroupedNvFp4MoeBf16>),
    Hybrid(Box<HybridNvFp4MoeBf16>),
    WeightOnly(Box<SelectedNvFp4MoeBf16>),
}

impl Candidate {
    pub(super) fn new(
        backend: &CudaBackend,
        request: MoePlanRequest,
        activation: GatedActivation,
        weights: &[NvFp4ExpertBank; 3],
        execution: MoeExecution,
    ) -> Result<Self> {
        let [gate, up, down] = weights;
        let plan = match execution {
            MoeExecution::DirectW4A4 => backend
                .prepare_direct_nvfp4_moe_bf16(
                    request.tokens,
                    request.top_k,
                    activation,
                    gate.clone(),
                    up.clone(),
                    down.clone(),
                )
                .map(Box::new)
                .map(Plan::Direct),
            MoeExecution::HybridW4A4 => backend
                .prepare_hybrid_gate_nvfp4_moe_bf16(
                    request.tokens,
                    request.top_k,
                    activation,
                    gate.clone(),
                    up.clone(),
                    down.clone(),
                )
                .map(Box::new)
                .map(Plan::Hybrid),
            MoeExecution::IndexedGrouped | MoeExecution::FusedIndexedGrouped => backend
                .prepare_grouped_nvfp4_moe_bf16(
                    request.tokens,
                    request.top_k,
                    activation,
                    execution,
                    gate.clone(),
                    up.clone(),
                    down.clone(),
                )
                .map(Box::new)
                .map(Plan::Grouped),
            MoeExecution::SelectedWeightOnly => backend
                .prepare_batched_selected_nvfp4_moe_bf16(
                    request.tokens,
                    request.top_k,
                    activation,
                    gate.clone(),
                    up.clone(),
                    down.clone(),
                )
                .map(Box::new)
                .map(Plan::WeightOnly),
            MoeExecution::Bucketed => backend
                .prepare_bucketed_nvfp4_moe_bf16(
                    request.tokens,
                    request.top_k,
                    activation,
                    gate.clone(),
                    up.clone(),
                    down.clone(),
                )
                .map(Box::new)
                .map(Plan::Bucketed),
        }?;
        Ok(Self { execution, plan })
    }
}

impl Plan {
    pub(super) fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        routing: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        match self {
            Self::Bucketed(plan) => plan.execute(input, selected, routing, output),
            Self::Direct(plan) => plan.execute(input, selected, routing, output),
            Self::Grouped(plan) => plan.execute(input, selected, routing, output),
            Self::Hybrid(plan) => plan.execute(input, selected, routing, output),
            Self::WeightOnly(plan) => plan.execute(input, selected, routing, output),
        }
    }
}
