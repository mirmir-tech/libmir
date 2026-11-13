use models::weights::{
    RoutedExpertBindings, SharedRoutedFeedForwardBindings, TensorCatalog, TensorStorage,
};

use self::{
    format::{merge_format, mixed_storage},
    storage::{routed_is_dense, routed_is_mxfp4, routed_is_mxfp8},
};
use super::AffineSharedExpertMoeConfig;
use crate::{
    AffineQuantizedWeight, CudaBackend, CudaTensorSet, Error, Result,
    backend::linear::{
        CheckpointProjectionWeight, DenseExpertWeights, MxFp4ExpertWeights, MxFp8ExpertWeights,
    },
};

mod format;
mod nvfp4;
mod storage;

#[derive(Clone, Debug)]
pub(super) enum RoutedSharedMoeWeights {
    Affine(Box<AffineRoutedMoeWeights>),
    Dense(Box<DenseExpertWeights>),
    MxFp4(Box<MxFp4ExpertWeights>),
    MxFp8(Box<MxFp8ExpertWeights>),
    NvFp4(Box<crate::backend::block::experts::ExpertWeights>),
}

#[derive(Clone, Debug)]
pub(super) struct AffineRoutedMoeWeights {
    pub(super) gate: AffineQuantizedWeight,
    pub(super) up: AffineQuantizedWeight,
    pub(super) down: AffineQuantizedWeight,
}

#[derive(Clone, Debug)]
pub struct AffineSharedExpertMoeWeights {
    pub(super) router: CheckpointProjectionWeight,
    pub(super) routed: RoutedSharedMoeWeights,
    pub(super) shared_gate: CheckpointProjectionWeight,
    pub(super) shared_up: CheckpointProjectionWeight,
    pub(super) shared_down: CheckpointProjectionWeight,
    pub(super) shared_output_gate: CheckpointProjectionWeight,
}

impl AffineSharedExpertMoeWeights {
    pub fn load(tensors: &CudaTensorSet, prefix: &str) -> Result<Self> {
        let affine = |name: &str| AffineQuantizedWeight::load(tensors, &format!("{prefix}.{name}"));
        Ok(Self {
            router: CheckpointProjectionWeight::Affine(affine("gate")?),
            routed: RoutedSharedMoeWeights::Affine(Box::new(AffineRoutedMoeWeights {
                gate: affine("switch_mlp.gate_proj")?,
                up: affine("switch_mlp.up_proj")?,
                down: affine("switch_mlp.down_proj")?,
            })),
            shared_gate: CheckpointProjectionWeight::Affine(affine("shared_expert.gate_proj")?),
            shared_up: CheckpointProjectionWeight::Affine(affine("shared_expert.up_proj")?),
            shared_down: CheckpointProjectionWeight::Affine(affine("shared_expert.down_proj")?),
            shared_output_gate: CheckpointProjectionWeight::Affine(affine("shared_expert_gate")?),
        })
    }

    pub fn load_bindings(
        backend: &CudaBackend,
        tensors: &CudaTensorSet,
        catalog: &TensorCatalog,
        bindings: SharedRoutedFeedForwardBindings<'_>,
        experts: usize,
        hidden: usize,
        intermediate: usize,
    ) -> Result<Self> {
        let routed = match bindings.routed {
            RoutedExpertBindings::SeparateGateUp { gate, up, down }
                if matches!(gate.storage, TensorStorage::AffineQuantized { .. })
                    && matches!(up.storage, TensorStorage::AffineQuantized { .. })
                    && matches!(down.storage, TensorStorage::AffineQuantized { .. }) =>
            {
                RoutedSharedMoeWeights::Affine(Box::new(AffineRoutedMoeWeights {
                    gate: AffineQuantizedWeight::load_binding(tensors, gate)?,
                    up: AffineQuantizedWeight::load_binding(tensors, up)?,
                    down: AffineQuantizedWeight::load_binding(tensors, down)?,
                }))
            },
            routed if routed_is_dense(routed) => RoutedSharedMoeWeights::Dense(Box::new(
                DenseExpertWeights::load(backend, tensors, routed, experts, hidden, intermediate)?,
            )),
            routed if routed_is_mxfp4(routed) => RoutedSharedMoeWeights::MxFp4(Box::new(
                MxFp4ExpertWeights::load(tensors, routed, experts, hidden, intermediate)?,
            )),
            routed if routed_is_mxfp8(routed) => RoutedSharedMoeWeights::MxFp8(Box::new(
                MxFp8ExpertWeights::load(tensors, routed, experts, hidden, intermediate)?,
            )),
            RoutedExpertBindings::Individual { .. } => RoutedSharedMoeWeights::NvFp4(Box::new(
                nvfp4::load(backend, catalog, bindings.routed, experts, hidden, intermediate)?,
            )),
            _ => {
                return Err(Error::UnsupportedDecoderLayer(
                    "CUDA shared-routed experts mix incompatible checkpoint storage".into(),
                ));
            },
        };
        Ok(Self {
            router: CheckpointProjectionWeight::load_binding_prepared(
                backend, tensors, bindings.router,
            )?,
            routed,
            shared_gate: CheckpointProjectionWeight::load_binding_prepared(
                backend,
                tensors,
                bindings.shared_gate,
            )?,
            shared_up: CheckpointProjectionWeight::load_binding_prepared(
                backend,
                tensors,
                bindings.shared_up,
            )?,
            shared_down: CheckpointProjectionWeight::load_binding_prepared(
                backend,
                tensors,
                bindings.shared_down,
            )?,
            shared_output_gate: CheckpointProjectionWeight::load_binding_prepared(
                backend,
                tensors,
                bindings.shared_output_gate,
            )?,
        })
    }

    pub(super) fn validate(&self, config: AffineSharedExpertMoeConfig) -> Result<()> {
        self.router.validate(
            1,
            config.hidden_size,
            config.expert_count,
            config.group_size,
            config.router_bits,
        )?;
        match &self.routed {
            RoutedSharedMoeWeights::Affine(weights) => {
                for weight in [&weights.gate, &weights.up] {
                    weight.validate(
                        config.expert_count,
                        config.hidden_size,
                        config.routed_intermediate_size,
                        config.group_size,
                        config.expert_bits,
                    )?;
                }
                weights.down.validate(
                    config.expert_count,
                    config.routed_intermediate_size,
                    config.hidden_size,
                    config.group_size,
                    config.expert_bits,
                )?;
            },
            RoutedSharedMoeWeights::Dense(weights) => {
                weights.spec(config.top_k, config.top_k, config.activation.into())?;
            },
            RoutedSharedMoeWeights::MxFp4(_)
            | RoutedSharedMoeWeights::MxFp8(_)
            | RoutedSharedMoeWeights::NvFp4(_) => {},
        }
        for weight in [&self.shared_gate, &self.shared_up] {
            weight.validate(
                1,
                config.hidden_size,
                config.shared_intermediate_size,
                config.group_size,
                config.expert_bits,
            )?;
        }
        self.shared_down.validate(
            1,
            config.shared_intermediate_size,
            config.hidden_size,
            config.group_size,
            config.expert_bits,
        )?;
        self.shared_output_gate.validate(
            1,
            config.hidden_size,
            1,
            config.group_size,
            config.router_bits,
        )
    }

    pub(in crate::backend) fn storage_format(
        &self,
        experts: usize,
        hidden: usize,
        routed: usize,
        shared: usize,
    ) -> Result<(usize, usize, usize)> {
        let mut expert_format = match &self.routed {
            RoutedSharedMoeWeights::Affine(weights) => {
                let expert = weights.gate.infer_config(experts, hidden, routed)?;
                Some((expert.group_size, expert.bits))
            },
            RoutedSharedMoeWeights::Dense(_)
            | RoutedSharedMoeWeights::MxFp4(_)
            | RoutedSharedMoeWeights::MxFp8(_)
            | RoutedSharedMoeWeights::NvFp4(_) => None,
        };
        for format in [
            self.shared_gate.affine_format(1, hidden, shared)?,
            self.shared_up.affine_format(1, hidden, shared)?,
            self.shared_down.affine_format(1, shared, hidden)?,
        ] {
            merge_format(&mut expert_format, format, "shared expert")?;
        }
        let mut router_format = self.router.affine_format(1, hidden, experts)?;
        merge_format(
            &mut router_format,
            self.shared_output_gate.affine_format(1, hidden, 1)?,
            "shared router",
        )?;
        if matches!(self.routed, RoutedSharedMoeWeights::Affine(_)) && router_format.is_none() {
            return Err(mixed_storage());
        }
        let group_size = match (expert_format, router_format) {
            (Some(expert), Some(control)) if expert.0 != control.0 => {
                return Err(mixed_storage());
            },
            (Some(format), _) | (_, Some(format)) => format.0,
            (None, None) => 0,
        };
        let expert_bits = expert_format.or(router_format).map_or(0, |format| format.1);
        let router_bits = router_format.or(expert_format).map_or(0, |format| format.1);
        Ok((group_size, expert_bits, router_bits))
    }
}
