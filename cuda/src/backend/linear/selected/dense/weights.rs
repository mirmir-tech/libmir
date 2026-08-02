use models::weights::{
    BindingTransform, HybridMoeExpertBindings, RoutedExpertBindings, TensorBinding, TensorStorage,
};

use super::canonical::canonicalize;
use crate::{
    CudaBackend, CudaTensor, CudaTensorDType, CudaTensorSet, Error, Result,
    kernels::{
        DenseExpertCanonicalizer, DenseGateUpLayout, DenseGatedActivation, SelectedDenseMoeSpec,
    },
};

#[derive(Clone, Debug)]
pub(in crate::backend) struct DenseProjectionWeight {
    pub weight: CudaTensor,
    pub bias: Option<CudaTensor>,
    pub transposed: bool,
}

#[derive(Clone, Debug)]
pub(in crate::backend) enum DenseGateUpWeights {
    Separate {
        gate: DenseProjectionWeight,
        up: DenseProjectionWeight,
    },
    Fused {
        projection: DenseProjectionWeight,
        interleaved: bool,
    },
}

#[derive(Clone, Debug)]
pub struct DenseExpertWeights {
    pub(in crate::backend) gate_up: DenseGateUpWeights,
    pub(in crate::backend) down: DenseProjectionWeight,
    experts: usize,
    hidden: usize,
    intermediate: usize,
}

impl DenseExpertWeights {
    pub(crate) fn load_hybrid(
        backend: &CudaBackend,
        tensors: &CudaTensorSet,
        bindings: &HybridMoeExpertBindings<'_>,
        experts: usize,
        hidden: usize,
        intermediate: usize,
    ) -> Result<Self> {
        let routed = match bindings {
            HybridMoeExpertBindings::Stacked(weights) => RoutedExpertBindings::SeparateGateUp {
                gate: weights.gate,
                up: weights.up,
                down: weights.down,
            },
            HybridMoeExpertBindings::FusedStacked { gate_up, down } => {
                RoutedExpertBindings::InterleavedGateUp { gate_up, down }
            },
            HybridMoeExpertBindings::Individual { .. } => {
                return Err(Error::UnsupportedDecoderLayer(
                    "dense CUDA experts require stacked checkpoint bindings".into(),
                ));
            },
        };
        Self::load(backend, tensors, routed, experts, hidden, intermediate)
    }

    pub(in crate::backend) fn load(
        backend: &CudaBackend,
        tensors: &CudaTensorSet,
        bindings: RoutedExpertBindings<'_>,
        experts: usize,
        hidden: usize,
        intermediate: usize,
    ) -> Result<Self> {
        let canonicalizer = DenseExpertCanonicalizer::compile(&backend.inner.compiler)?;
        let (gate_up, down) = match bindings {
            RoutedExpertBindings::SeparateGateUp { gate, up, down } => (
                DenseGateUpWeights::Separate {
                    gate: projection(
                        backend, &canonicalizer, tensors, gate, experts, hidden, intermediate,
                    )?,
                    up: projection(
                        backend, &canonicalizer, tensors, up, experts, hidden, intermediate,
                    )?,
                },
                projection(backend, &canonicalizer, tensors, down, experts, intermediate, hidden)?,
            ),
            RoutedExpertBindings::InterleavedGateUp { gate_up, down } => (
                DenseGateUpWeights::Fused {
                    projection: projection(
                        backend,
                        &canonicalizer,
                        tensors,
                        gate_up,
                        experts,
                        hidden,
                        intermediate.checked_mul(2).ok_or(Error::InvalidDecoderKernel(
                            "dense fused expert width overflow",
                        ))?,
                    )?,
                    interleaved: gate_up
                        .transforms
                        .contains(&BindingTransform::FusedGateUp { interleaved: true }),
                },
                projection(backend, &canonicalizer, tensors, down, experts, intermediate, hidden)?,
            ),
            RoutedExpertBindings::Individual { .. } => {
                return Err(Error::UnsupportedDecoderLayer(
                    "dense CUDA experts require stacked checkpoint bindings".into(),
                ));
            },
        };
        Ok(Self {
            gate_up,
            down,
            experts,
            hidden,
            intermediate,
        })
    }

    pub(in crate::backend) fn spec(
        &self,
        tokens: usize,
        selected_count: usize,
        activation: DenseGatedActivation,
    ) -> Result<SelectedDenseMoeSpec> {
        let (layout, gate, up) = match &self.gate_up {
            DenseGateUpWeights::Separate { gate, up } => (DenseGateUpLayout::Separate, gate, up),
            DenseGateUpWeights::Fused { projection, interleaved } => (
                if *interleaved {
                    DenseGateUpLayout::FusedInterleaved
                } else {
                    DenseGateUpLayout::FusedContiguous
                },
                projection,
                projection,
            ),
        };
        SelectedDenseMoeSpec {
            tokens,
            input_features: self.hidden,
            output_features: self.intermediate,
            expert_count: self.experts,
            selected_count,
            gate_up_layout: layout,
            gate_transposed: gate.transposed,
            up_transposed: up.transposed,
            down_transposed: self.down.transposed,
            gate_bias: gate.bias.is_some(),
            up_bias: up.bias.is_some(),
            down_bias: self.down.bias.is_some(),
            activation,
        }
        .validate()
    }

    pub(in crate::backend) fn gate_up(&self) -> (&DenseProjectionWeight, &DenseProjectionWeight) {
        match &self.gate_up {
            DenseGateUpWeights::Separate { gate, up } => (gate, up),
            DenseGateUpWeights::Fused { projection, .. } => (projection, projection),
        }
    }

    pub(in crate::backend) fn intermediate_elements(
        &self,
        tokens: usize,
        selected: usize,
    ) -> Result<usize> {
        tokens
            .checked_mul(selected)
            .and_then(|value| value.checked_mul(self.intermediate))
            .ok_or(Error::InvalidDecoderKernel("dense selected-expert scratch size overflow"))
    }
}

fn projection(
    backend: &CudaBackend,
    canonicalizer: &DenseExpertCanonicalizer,
    tensors: &CudaTensorSet,
    binding: &TensorBinding,
    experts: usize,
    input: usize,
    output: usize,
) -> Result<DenseProjectionWeight> {
    let TensorStorage::Dense { bias, .. } = &binding.storage else {
        return Err(Error::UnsupportedDecoderLayer(format!(
            "dense selected expert requires dense storage: {}",
            binding.source
        )));
    };
    let transposed = binding.transforms.contains(&BindingTransform::Transpose);
    let expected = if transposed {
        vec![experts, input, output]
    } else {
        vec![experts, output, input]
    };
    let source = required(tensors, &binding.source)?;
    validate(&source, &expected)?;
    let weight = if transposed {
        canonicalize(backend, canonicalizer, &source, experts, input, output)?
    } else {
        source
    };
    let bias = bias
        .as_deref()
        .map(|name| {
            let tensor = required(tensors, name)?;
            validate(&tensor, &[experts, output])?;
            Ok::<CudaTensor, Error>(tensor)
        })
        .transpose()?;
    Ok(DenseProjectionWeight { weight, bias, transposed: false })
}

fn required(tensors: &CudaTensorSet, name: &str) -> Result<CudaTensor> {
    tensors.get(name).cloned().ok_or_else(|| Error::MissingTensor(name.into()))
}

fn validate(tensor: &CudaTensor, expected: &[usize]) -> Result<()> {
    if tensor.dtype() != CudaTensorDType::Bf16 {
        return Err(Error::DTypeMismatch {
            name: tensor.name().into(),
            expected: "BF16",
        });
    }
    if tensor.shape() != expected {
        return Err(Error::InvalidQuantizedTensor {
            name: tensor.name().into(),
            expected: expected.to_vec(),
            actual: tensor.shape().to_vec(),
        });
    }
    Ok(())
}
