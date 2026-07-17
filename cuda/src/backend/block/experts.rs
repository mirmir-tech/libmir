use mircuda::{DeviceBuffer, bf16};

use crate::{
    BucketedNvFp4MoeBf16, CudaBackend, DirectNvFp4MoeBf16, GatedActivation, GroupedNvFp4MoeBf16,
    HybridNvFp4MoeBf16, MoeExecution, NvFp4ExpertBank, Result, SelectedNvFp4MoeBf16,
};

#[derive(Debug)]
pub(super) enum Experts {
    Bucketed(Box<BucketedNvFp4MoeBf16>),
    Direct(Box<DirectNvFp4MoeBf16>),
    Grouped(Box<GroupedNvFp4MoeBf16>),
    Hybrid(Box<HybridNvFp4MoeBf16>),
    WeightOnly(Box<SelectedNvFp4MoeBf16>),
}

impl Experts {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        backend: &CudaBackend,
        tokens: usize,
        selected: usize,
        activation: GatedActivation,
        execution: MoeExecution,
        gate: NvFp4ExpertBank,
        up: NvFp4ExpertBank,
        down: NvFp4ExpertBank,
    ) -> Result<Self> {
        match execution {
            MoeExecution::DirectW4A4 => backend
                .prepare_direct_nvfp4_moe_bf16(tokens, selected, activation, gate, up, down)
                .map(Box::new)
                .map(Self::Direct),
            MoeExecution::HybridW4A4 => backend
                .prepare_hybrid_gate_nvfp4_moe_bf16(tokens, selected, activation, gate, up, down)
                .map(Box::new)
                .map(Self::Hybrid),
            MoeExecution::IndexedGrouped | MoeExecution::FusedIndexedGrouped => backend
                .prepare_grouped_nvfp4_moe_bf16(
                    tokens, selected, activation, execution, gate, up, down,
                )
                .map(Box::new)
                .map(Self::Grouped),
            MoeExecution::SelectedWeightOnly => backend
                .prepare_batched_selected_nvfp4_moe_bf16(
                    tokens, selected, activation, gate, up, down,
                )
                .map(Box::new)
                .map(Self::WeightOnly),
            MoeExecution::Bucketed => backend
                .prepare_bucketed_nvfp4_moe_bf16(tokens, selected, activation, gate, up, down)
                .map(Box::new)
                .map(Self::Bucketed),
        }
    }

    pub(super) fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        routing: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        match self {
            Self::Bucketed(experts) => experts.execute(input, selected, routing, output),
            Self::Direct(experts) => experts.execute(input, selected, routing, output),
            Self::Grouped(experts) => experts.execute(input, selected, routing, output),
            Self::Hybrid(experts) => experts.execute(input, selected, routing, output),
            Self::WeightOnly(experts) => experts.execute(input, selected, routing, output),
        }
    }
}
