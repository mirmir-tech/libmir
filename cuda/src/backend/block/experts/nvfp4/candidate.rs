use mircuda::{DeviceBuffer, MarlinNvFp4ThreadConfig, bf16};

use crate::{
    BucketedNvFp4MoeBf16, CudaBackend, DirectNvFp4MoeBf16, GatedActivation, GroupedNvFp4MoeBf16,
    HybridNvFp4MoeBf16, MoeExecution, MoePlanRequest, NvFp4ExpertBank, Result,
    SelectedNvFp4MoeBf16,
    backend::linear::{
        MarlinNvFp4MoeBf16, SelectedNvFp4WeightOnlyTensorCoreMoeBf16, TiledSelectedNvFp4MoeBf16,
    },
    kernels::SelectedNvFp4TiledRows,
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
    TensorCoreWeightOnly(Box<SelectedNvFp4WeightOnlyTensorCoreMoeBf16>),
    TiledWeightOnly(Box<TiledSelectedNvFp4MoeBf16>),
    MarlinWeightOnly(Box<MarlinNvFp4MoeBf16>),
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
            MoeExecution::SelectedWeightOnlyTensorCore => backend
                .prepare_selected_nvfp4_weight_only_tensor_core_moe_bf16(
                    request.tokens,
                    request.top_k,
                    activation,
                    [gate.clone(), up.clone(), down.clone()],
                )
                .map(Box::new)
                .map(Plan::TensorCoreWeightOnly),
            MoeExecution::SelectedWeightOnlyTiled2
            | MoeExecution::SelectedWeightOnlyTiled4
            | MoeExecution::SelectedWeightOnlyTiled8 => {
                let rows = match execution {
                    MoeExecution::SelectedWeightOnlyTiled2 => SelectedNvFp4TiledRows::Two,
                    MoeExecution::SelectedWeightOnlyTiled4 => SelectedNvFp4TiledRows::Four,
                    MoeExecution::SelectedWeightOnlyTiled8 => SelectedNvFp4TiledRows::Eight,
                    _ => unreachable!(),
                };
                backend
                    .prepare_tiled_selected_nvfp4_moe_bf16(
                        request.tokens,
                        request.top_k,
                        activation,
                        rows,
                        [gate.clone(), up.clone(), down.clone()],
                    )
                    .map(Box::new)
                    .map(Plan::TiledWeightOnly)
            },
            MoeExecution::MarlinWeightOnlyN128K128
            | MoeExecution::MarlinWeightOnlyN128K64
            | MoeExecution::MarlinWeightOnlyN64K128 => prepare_marlin(
                backend,
                request,
                activation,
                [gate.clone(), up.clone(), down.clone()],
                execution,
            ),
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

fn prepare_marlin(
    backend: &CudaBackend,
    request: MoePlanRequest,
    activation: GatedActivation,
    banks: [NvFp4ExpertBank; 3],
    execution: MoeExecution,
) -> Result<Plan> {
    let thread_config = match execution {
        MoeExecution::MarlinWeightOnlyN128K128 => MarlinNvFp4ThreadConfig::N128K128,
        MoeExecution::MarlinWeightOnlyN128K64 => MarlinNvFp4ThreadConfig::N128K64,
        MoeExecution::MarlinWeightOnlyN64K128 => MarlinNvFp4ThreadConfig::N64K128,
        _ => unreachable!(),
    };
    backend
        .prepare_marlin_nvfp4_moe_bf16(
            request.tokens, request.top_k, activation, thread_config, banks,
        )
        .map(Box::new)
        .map(Plan::MarlinWeightOnly)
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
            Self::TensorCoreWeightOnly(plan) => plan.execute(input, selected, routing, output),
            Self::TiledWeightOnly(plan) => plan.execute(input, selected, routing, output),
            Self::MarlinWeightOnly(plan) => plan.execute(input, selected, routing, output),
        }
    }

    pub(super) fn prequant_scale(&self) -> Option<DeviceBuffer<f32>> {
        match self {
            Self::Bucketed(plan) => plan.prequant_scale(),
            _ => None,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn execute_prequantized_residual_shared(
        &mut self,
        packed: &DeviceBuffer<u8>,
        scales: &DeviceBuffer<u8>,
        selected: &DeviceBuffer<u32>,
        routing: &DeviceBuffer<bf16>,
        residual: &DeviceBuffer<bf16>,
        shared: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let Self::Bucketed(plan) = self else {
            return Err(crate::Error::InvalidExecutionPlan(
                "NVFP4 expert plan cannot fuse routed output",
            ));
        };
        plan.execute_prequantized_residual_shared(
            packed, scales, selected, routing, residual, shared, output,
        )
    }
}
