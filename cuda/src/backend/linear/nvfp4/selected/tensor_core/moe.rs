use mircuda::{DeviceBuffer, Stream, bf16};

use super::{
    super::{CudaBackend, GatedActivation, NvFp4ExpertBank},
    SelectedNvFp4LinearBf16,
};
use crate::{Error, Result, kernels::ElementwiseBf16};

/// Complete device-selected NVFP4 MLP using native W4A4 Tensor Cores.
#[derive(Debug)]
pub struct SelectedNvFp4TensorCoreMoeBf16 {
    gate: SelectedNvFp4LinearBf16,
    up: SelectedNvFp4LinearBf16,
    down: SelectedNvFp4LinearBf16,
    gated: ElementwiseBf16,
    reduce: ElementwiseBf16,
    gate_output: DeviceBuffer<bf16>,
    up_output: DeviceBuffer<bf16>,
    intermediate: DeviceBuffer<bf16>,
    down_output: DeviceBuffer<bf16>,
    activation: GatedActivation,
    stream: Stream,
}

impl CudaBackend {
    /// Prepares gate, up, activation, down, and router-weighted W4A4 execution.
    pub fn prepare_selected_nvfp4_tensor_core_moe_bf16(
        &self,
        selected: usize,
        activation: GatedActivation,
        gate: NvFp4ExpertBank,
        up: NvFp4ExpertBank,
        down: NvFp4ExpertBank,
    ) -> Result<SelectedNvFp4TensorCoreMoeBf16> {
        SelectedNvFp4TensorCoreMoeBf16::new(self, selected, activation, gate, up, down)
    }
}

impl SelectedNvFp4TensorCoreMoeBf16 {
    fn new(
        backend: &CudaBackend,
        selected: usize,
        activation: GatedActivation,
        gate_bank: NvFp4ExpertBank,
        up_bank: NvFp4ExpertBank,
        down_bank: NvFp4ExpertBank,
    ) -> Result<Self> {
        let gate = SelectedNvFp4LinearBf16::new(backend, selected, gate_bank)?;
        let up = SelectedNvFp4LinearBf16::new(backend, selected, up_bank)?;
        let down = SelectedNvFp4LinearBf16::new(backend, selected, down_bank)?;
        validate(&gate, &up, &down)?;
        let intermediate_elements = gate.output_elements()?;
        let output_elements = down.output_elements()?;
        let allocate =
            |elements| backend.inner.pool.allocate::<bf16>(&backend.inner.stream, elements);
        Ok(Self {
            gated: ElementwiseBf16::compile(&backend.inner.compiler, intermediate_elements)?,
            reduce: ElementwiseBf16::compile(&backend.inner.compiler, down.output_features())?,
            gate_output: allocate(intermediate_elements)?,
            up_output: allocate(intermediate_elements)?,
            intermediate: allocate(intermediate_elements)?,
            down_output: allocate(output_elements)?,
            gate,
            up,
            down,
            activation,
            stream: backend.inner.stream.clone(),
        })
    }

    /// Enqueues the complete routed MLP without allocation or host
    /// synchronization.
    pub fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        routing_weights: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.gate.execute(input, selected, &mut self.gate_output)?;
        self.up.execute(input, selected, &mut self.up_output)?;
        self.gated.gated(
            &self.stream,
            &self.gate_output,
            &self.up_output,
            &mut self.intermediate,
            self.activation.into(),
        )?;
        self.down.execute_ranked(&self.intermediate, selected, &mut self.down_output)?;
        self.reduce.weighted_reduce(
            &self.stream,
            &self.down_output,
            routing_weights,
            output,
            self.down.selected_count(),
        )
    }

    #[must_use]
    pub const fn output_elements(&self) -> usize {
        self.down.output_features()
    }
}

fn validate(
    gate: &SelectedNvFp4LinearBf16,
    up: &SelectedNvFp4LinearBf16,
    down: &SelectedNvFp4LinearBf16,
) -> Result<()> {
    let hidden = gate.input_features();
    let intermediate = gate.output_features();
    let input_geometry = up.input_features() == hidden && down.output_features() == hidden;
    let intermediate_geometry =
        up.output_features() == intermediate && down.input_features() == intermediate;
    if gate.selected_count() != up.selected_count()
        || gate.selected_count() != down.selected_count()
        || !input_geometry
        || !intermediate_geometry
    {
        Err(Error::InvalidNvFp4("incompatible Tensor Core expert banks"))
    } else {
        Ok(())
    }
}
