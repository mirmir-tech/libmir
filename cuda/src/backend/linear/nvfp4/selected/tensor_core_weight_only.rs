use mircuda::{DeviceBuffer, Stream, bf16};

use super::{CudaBackend, NvFp4ExpertBank, view};
use crate::{
    Error, GatedActivation, Result,
    kernels::{
        ElementwiseBf16, SelectedNvFp4Spec, SelectedNvFp4TensorCoreGated,
        SelectedNvFp4TensorCoreLinear,
    },
};

/// Selected-expert W4A16 candidate using BF16 Tensor Core accumulation.
#[derive(Debug)]
pub struct SelectedNvFp4WeightOnlyTensorCoreMoeBf16 {
    gated: SelectedNvFp4TensorCoreGated,
    down: SelectedNvFp4TensorCoreLinear,
    reduce: ElementwiseBf16,
    gate: NvFp4ExpertBank,
    up: NvFp4ExpertBank,
    down_bank: NvFp4ExpertBank,
    intermediate: DeviceBuffer<bf16>,
    down_output: DeviceBuffer<bf16>,
    stream: Stream,
    tokens: usize,
    selected: usize,
}

impl SelectedNvFp4WeightOnlyTensorCoreMoeBf16 {
    pub(super) fn new(
        backend: &CudaBackend,
        tokens: usize,
        selected: usize,
        activation: GatedActivation,
        banks: [NvFp4ExpertBank; 3],
    ) -> Result<Self> {
        let [gate, up, down_bank] = banks;
        validate(tokens, selected, &gate, &up, &down_bank)?;
        let spec = SelectedNvFp4Spec {
            experts: gate.config.experts,
            selected,
            hidden: gate.config.input_features,
            intermediate: gate.config.output_features,
            activation: activation.into(),
        };
        let routes = tokens
            .checked_mul(selected)
            .ok_or(Error::InvalidNvFp4("Tensor Core selected route overflow"))?;
        let intermediate_elements = routes
            .checked_mul(spec.intermediate)
            .ok_or(Error::InvalidNvFp4("Tensor Core intermediate overflow"))?;
        let down_elements = routes
            .checked_mul(spec.hidden)
            .ok_or(Error::InvalidNvFp4("Tensor Core down output overflow"))?;
        let allocate = |elements| backend.inner.pool.allocate(&backend.inner.stream, elements);
        Ok(Self {
            gated: SelectedNvFp4TensorCoreGated::compile(&backend.inner.compiler, spec)?,
            down: SelectedNvFp4TensorCoreLinear::compile(&backend.inner.compiler, spec)?,
            reduce: ElementwiseBf16::compile(&backend.inner.compiler, spec.hidden)?,
            intermediate: allocate(intermediate_elements)?,
            down_output: allocate(down_elements)?,
            gate,
            up,
            down_bank,
            stream: backend.inner.stream.clone(),
            tokens,
            selected,
        })
    }

    pub fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        routing: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.gated.execute(
            &self.stream,
            input,
            selected,
            view(&self.gate),
            view(&self.up),
            &mut self.intermediate,
            self.tokens,
        )?;
        self.down.execute(
            &self.stream,
            &self.intermediate,
            selected,
            view(&self.down_bank),
            &mut self.down_output,
            self.tokens,
        )?;
        self.reduce.weighted_reduce_batch(
            &self.stream,
            &self.down_output,
            routing,
            output,
            self.selected,
            self.tokens,
        )
    }
}

fn validate(
    tokens: usize,
    selected: usize,
    gate: &NvFp4ExpertBank,
    up: &NvFp4ExpertBank,
    down: &NvFp4ExpertBank,
) -> Result<()> {
    let valid = tokens > 0
        && selected > 0
        && selected <= gate.config.experts
        && gate.config == up.config
        && (gate.config.input_features, gate.config.output_features, gate.config.experts)
            == (down.config.output_features, down.config.input_features, down.config.experts);
    if valid {
        Ok(())
    } else {
        Err(Error::InvalidNvFp4("incompatible W4A16 Tensor Core expert banks"))
    }
}
