use mircuda::{DeviceBuffer, Stream, bf16};
use models::layout::DecoderConfig;

use super::{AffineQuantizedPairTensors, u32_tensor, validate_bank};
use crate::{
    CudaBackend, Error, Result,
    backend::linear::quantized::{bf16_tensor, expected_shape},
    kernels::{
        AffineGemvSpec, SelectedAffineGated, SelectedAffineGatedLaunch, SelectedAffineGatedSpec,
    },
};

/// Gated MLP activation derived from model metadata.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GatedActivation {
    GeluTanh,
    Silu,
}

impl GatedActivation {
    pub(crate) const fn code(self) -> u32 {
        match self {
            Self::GeluTanh => 0,
            Self::Silu => 1,
        }
    }
}

impl TryFrom<&str> for GatedActivation {
    type Error = Error;

    fn try_from(value: &str) -> Result<Self> {
        match value {
            "gelu_pytorch_tanh" | "gelu_new" => Ok(Self::GeluTanh),
            "silu" | "swish" => Ok(Self::Silu),
            value => Err(Error::UnsupportedGatedActivation(value.into())),
        }
    }
}

impl TryFrom<&DecoderConfig> for GatedActivation {
    type Error = Error;

    fn try_from(config: &DecoderConfig) -> Result<Self> {
        config
            .hidden_activation
            .as_deref()
            .ok_or_else(|| Error::UnsupportedGatedActivation("<missing>".into()))?
            .try_into()
    }
}

impl From<GatedActivation> for crate::kernels::GatedActivation {
    fn from(value: GatedActivation) -> Self {
        match value {
            GatedActivation::GeluTanh => Self::GeluTanh,
            GatedActivation::Silu => Self::Silu,
        }
    }
}

/// Fused selected gate/up projections and gated activation.
#[derive(Clone, Debug)]
pub struct SelectedAffineGatedBf16Linear {
    operation: SelectedAffineGated,
    stream: Stream,
}

impl SelectedAffineGatedBf16Linear {
    pub(in crate::backend) fn new(
        backend: &CudaBackend,
        matrix: AffineGemvSpec,
        expert_count: usize,
        selected_count: usize,
        activation: GatedActivation,
    ) -> Result<Self> {
        Self::new_batch(backend, matrix, expert_count, selected_count, 1, activation)
    }

    pub(in crate::backend) fn new_batch(
        backend: &CudaBackend,
        matrix: AffineGemvSpec,
        expert_count: usize,
        selected_count: usize,
        tokens: usize,
        activation: GatedActivation,
    ) -> Result<Self> {
        let spec = SelectedAffineGatedSpec::new_batch(
            matrix,
            expert_count,
            selected_count,
            tokens,
            activation.into(),
        )?;
        Ok(Self {
            operation: SelectedAffineGated::compile(&backend.inner.compiler, spec)?,
            stream: backend.inner.stream.clone(),
        })
    }

    /// Enqueues selected gate/up projections and activation in one launch.
    pub fn execute(
        &self,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        tensors: AffineQuantizedPairTensors<'_>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let spec = self.operation.spec();
        let matrix = spec.matrix;
        let packed = matrix.input_features / (32 / matrix.bits);
        let groups = matrix.input_features / matrix.group_size;
        let weight_shape = expected_shape(spec.expert_count, matrix.output_features, packed);
        let group_shape = expected_shape(spec.expert_count, matrix.output_features, groups);
        validate_bank(tensors.gate, &weight_shape, &group_shape)?;
        validate_bank(tensors.up, &weight_shape, &group_shape)?;
        self.operation.execute(
            &self.stream,
            &mut SelectedAffineGatedLaunch {
                input,
                selected,
                gate_weight: u32_tensor(tensors.gate.weight)?,
                gate_scales: bf16_tensor(tensors.gate.scales)?,
                gate_biases: bf16_tensor(tensors.gate.biases)?,
                up_weight: u32_tensor(tensors.up.weight)?,
                up_scales: bf16_tensor(tensors.up.scales)?,
                up_biases: bf16_tensor(tensors.up.biases)?,
                output,
            },
        )
    }

    /// Elements required by the selected intermediate output.
    pub fn output_elements(&self) -> Result<usize> {
        let spec = self.operation.spec();
        spec.matrix
            .output_features
            .checked_mul(spec.selected_count)
            .and_then(|elements| elements.checked_mul(spec.tokens))
            .ok_or_else(|| Error::InvalidTensorSize {
                name: "selected gated output".into(),
                expected: usize::MAX,
                actual: 0,
            })
    }
}
