use models::weights::RoutedExpertBindings;

use super::{
    super::{
        expert_tuning,
        fused_gate_up::{split_interleaved_last, split_last},
    },
    Array, BoundLinear, Error, FusedExpertGateUp, ModelTensors, Result, Stream,
};

#[derive(Debug)]
pub(super) enum RoutedGateUp {
    Separate {
        gate: BoundLinear,
        up: BoundLinear,
        fused: Option<Box<FusedExpertGateUp>>,
    },
    Fused {
        projection: BoundLinear,
        width: usize,
        interleaved: bool,
    },
}

impl RoutedGateUp {
    pub(super) fn load(
        tensors: &ModelTensors,
        bindings: RoutedExpertBindings<'_>,
        stream: &Stream,
    ) -> Result<(Self, BoundLinear)> {
        match bindings {
            RoutedExpertBindings::SeparateGateUp { gate, up, down } => Ok((
                Self::Separate {
                    gate: BoundLinear::load(tensors, gate, stream)?,
                    up: BoundLinear::load(tensors, up, stream)?,
                    fused: None,
                },
                BoundLinear::load(tensors, down, stream)?,
            )),
            RoutedExpertBindings::InterleavedGateUp { gate_up, down } => {
                let output = gate_up.shape.get(1).copied().ok_or(Error::ShapeOverflow)?;
                if !output.is_multiple_of(2) {
                    return Err(Error::InvalidModel(
                        "fused expert gate/up width must be even".into(),
                    ));
                }
                Ok((
                    Self::Fused {
                        projection: BoundLinear::load(tensors, gate_up, stream)?,
                        width: output / 2,
                        interleaved: gate_up.transforms.contains(
                            &models::weights::BindingTransform::FusedGateUp { interleaved: true },
                        ),
                    },
                    BoundLinear::load(tensors, down, stream)?,
                ))
            },
            RoutedExpertBindings::Individual { .. } => Err(Error::InvalidModel(
                "shared routed Metal execution requires stacked expert tensors".into(),
            )),
        }
    }

    pub(super) fn enable(&mut self, stream: &Stream) -> Result<bool> {
        match self {
            Self::Separate { gate, up, fused } => {
                if fused.is_none() {
                    *fused = gate.fuse_expert_gate_up(up, stream)?.map(Box::new);
                }
                fused.as_deref().map_or(Ok(()), |fused| fused.warm(stream))?;
                Ok(fused.is_some())
            },
            Self::Fused { .. } => Ok(true),
        }
    }

    pub(super) fn fused_bytes(&self) -> Result<Option<usize>> {
        match self {
            Self::Separate { gate, up, .. } => gate.fused_expert_gate_up_bytes(up),
            Self::Fused { .. } => Ok(Some(0)),
        }
    }

    pub(super) const fn is_fused(&self) -> bool {
        match self {
            Self::Separate { fused, .. } => fused.is_some(),
            Self::Fused { .. } => true,
        }
    }

    pub(super) fn tuning_format(&self) -> (i32, i32) {
        match self {
            Self::Separate { gate, .. } => gate.tuning_format(),
            Self::Fused { projection, .. } => projection.tuning_format(),
        }
    }

    pub(super) fn gather(
        &self,
        input: &Array,
        indices: &Array,
        sorted: bool,
        stream: &Stream,
    ) -> Result<(Array, Array)> {
        match self {
            Self::Separate { gate, up, fused } => {
                let fused = (!sorted).then_some(fused.as_deref()).flatten();
                fused.map_or_else(
                    || {
                        Ok((
                            gate.gather(input, indices, sorted, stream)?,
                            up.gather(input, indices, sorted, stream)?,
                        ))
                    },
                    |projection| {
                        expert_tuning::forward(gate, up, projection, input, indices, stream)
                    },
                )
            },
            Self::Fused { projection, width, interleaved } => {
                let output = projection.gather(input, indices, sorted, stream)?;
                if *interleaved {
                    split_interleaved_last(&output, *width, stream)
                } else {
                    split_last(&output, *width, stream)
                }
            },
        }
    }

    pub(super) fn gather_native(
        &self,
        input: &Array,
        indices: &Array,
        stream: &Stream,
    ) -> Result<(Array, Array)> {
        match self {
            Self::Separate { gate, up, fused: None } => Ok((
                gate.gather_native(input, indices, false, stream)?,
                up.gather_native(input, indices, false, stream)?,
            )),
            Self::Fused { projection, width, interleaved } => {
                let output = projection.gather_native(input, indices, false, stream)?;
                if *interleaved {
                    split_interleaved_last(&output, *width, stream)
                } else {
                    split_last(&output, *width, stream)
                }
            },
            Self::Separate { .. } => self.gather(input, indices, false, stream),
        }
    }
}
