use mircuda::{DeviceBuffer, Stream, bf16};

use super::{
    super::{CudaBackend, NvFp4ExpertBank},
    GroupedNvFp4LinearBf16, GroupedNvFp4PairBf16,
};
use crate::{Error, GatedActivation, MoeExecution, Result, kernels::ElementwiseBf16};

#[derive(Debug)]
enum ActivationStage {
    Split {
        operation: ElementwiseBf16,
        intermediate: DeviceBuffer<bf16>,
    },
    Fused,
}

/// Batched device-routed NVFP4 MLP using indexed grouped W4A4 CUTLASS.
#[derive(Debug)]
pub struct GroupedNvFp4MoeBf16 {
    gate_up: GroupedNvFp4PairBf16,
    down: GroupedNvFp4LinearBf16,
    activation_stage: ActivationStage,
    reduce: ElementwiseBf16,
    gate_output: DeviceBuffer<bf16>,
    up_output: DeviceBuffer<bf16>,
    down_output: DeviceBuffer<bf16>,
    activation: GatedActivation,
    stream: Stream,
    tokens: usize,
    selected: usize,
    output_elements: usize,
}

impl CudaBackend {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::backend) fn prepare_grouped_nvfp4_moe_bf16(
        &self,
        tokens: usize,
        selected: usize,
        activation: GatedActivation,
        execution: MoeExecution,
        gate: NvFp4ExpertBank,
        up: NvFp4ExpertBank,
        down: NvFp4ExpertBank,
    ) -> Result<GroupedNvFp4MoeBf16> {
        GroupedNvFp4MoeBf16::new(self, tokens, selected, activation, execution, gate, up, down)
    }
}

impl GroupedNvFp4MoeBf16 {
    #[allow(clippy::too_many_arguments)]
    fn new(
        backend: &CudaBackend,
        tokens: usize,
        selected: usize,
        activation: GatedActivation,
        execution: MoeExecution,
        gate_bank: NvFp4ExpertBank,
        up_bank: NvFp4ExpertBank,
        down_bank: NvFp4ExpertBank,
    ) -> Result<Self> {
        let gate_up = GroupedNvFp4PairBf16::new(backend, tokens, selected, gate_bank, up_bank)?;
        let down = GroupedNvFp4LinearBf16::new(backend, tokens, selected, down_bank)?;
        let intermediate_elements = gate_up.output_elements()?;
        if down.output_features() == 0 || gate_up.output_features() != down.input_features() {
            return Err(Error::InvalidNvFp4("incompatible grouped expert banks"));
        }
        let down_elements = down.output_elements()?;
        let output_elements = tokens
            .checked_mul(down.output_features())
            .ok_or(Error::InvalidNvFp4("grouped MoE output overflow"))?;
        let allocate = |elements| backend.inner.pool.allocate(&backend.inner.stream, elements);
        let activation_stage = match execution {
            MoeExecution::IndexedGrouped => ActivationStage::Split {
                operation: ElementwiseBf16::compile(
                    &backend.inner.compiler,
                    intermediate_elements,
                )?,
                intermediate: allocate(intermediate_elements)?,
            },
            MoeExecution::FusedIndexedGrouped => ActivationStage::Fused,
            MoeExecution::DirectW4A4
            | MoeExecution::HybridW4A4
            | MoeExecution::SelectedWeightOnly
            | MoeExecution::SelectedWeightOnlyTensorCore
            | MoeExecution::SelectedWeightOnlyTiled2
            | MoeExecution::SelectedWeightOnlyTiled4
            | MoeExecution::SelectedWeightOnlyTiled8
            | MoeExecution::MarlinWeightOnlyN128K128
            | MoeExecution::MarlinWeightOnlyN128K64
            | MoeExecution::MarlinWeightOnlyN64K128
            | MoeExecution::Bucketed => {
                return Err(Error::InvalidExecutionPlan(
                    "decode MoE cannot use bucketed execution",
                ));
            },
        };
        Ok(Self {
            activation_stage,
            reduce: ElementwiseBf16::compile(&backend.inner.compiler, down.output_features())?,
            gate_output: allocate(intermediate_elements)?,
            up_output: allocate(intermediate_elements)?,
            down_output: allocate(down_elements)?,
            gate_up,
            down,
            activation,
            stream: backend.inner.stream.clone(),
            tokens,
            selected,
            output_elements,
        })
    }

    pub fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        routing: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.gate_up
            .execute(input, selected, &mut self.gate_output, &mut self.up_output)?;
        match &mut self.activation_stage {
            ActivationStage::Split { operation, intermediate } => {
                operation.gated(
                    &self.stream,
                    &self.gate_output,
                    &self.up_output,
                    intermediate,
                    self.activation.into(),
                )?;
                self.down.execute_ranked(intermediate, selected, &mut self.down_output)?;
            },
            ActivationStage::Fused => {
                self.down.quantize_gated(
                    &self.gate_output,
                    &self.up_output,
                    selected,
                    self.activation,
                )?;
                self.down.execute_prepared(selected, &mut self.down_output)?;
            },
        }
        self.reduce.weighted_reduce_batch(
            &self.stream,
            &self.down_output,
            routing,
            output,
            self.selected,
            self.tokens,
        )
    }

    #[must_use]
    pub const fn output_elements(&self) -> usize {
        self.output_elements
    }
}
