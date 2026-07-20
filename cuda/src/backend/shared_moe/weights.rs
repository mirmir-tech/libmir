use super::AffineSharedExpertMoeConfig;
use crate::{AffineQuantizedWeight, CudaTensorSet, Result};

#[derive(Clone, Debug)]
pub struct AffineSharedExpertMoeWeights {
    pub router: AffineQuantizedWeight,
    pub routed_gate: AffineQuantizedWeight,
    pub routed_up: AffineQuantizedWeight,
    pub routed_down: AffineQuantizedWeight,
    pub shared_gate: AffineQuantizedWeight,
    pub shared_up: AffineQuantizedWeight,
    pub shared_down: AffineQuantizedWeight,
    pub shared_output_gate: AffineQuantizedWeight,
}

impl AffineSharedExpertMoeWeights {
    pub fn load(tensors: &CudaTensorSet, prefix: &str) -> Result<Self> {
        let load = |name: &str| AffineQuantizedWeight::load(tensors, &format!("{prefix}.{name}"));
        Ok(Self {
            router: load("gate")?,
            routed_gate: load("switch_mlp.gate_proj")?,
            routed_up: load("switch_mlp.up_proj")?,
            routed_down: load("switch_mlp.down_proj")?,
            shared_gate: load("shared_expert.gate_proj")?,
            shared_up: load("shared_expert.up_proj")?,
            shared_down: load("shared_expert.down_proj")?,
            shared_output_gate: load("shared_expert_gate")?,
        })
    }

    pub(super) fn validate(&self, config: AffineSharedExpertMoeConfig) -> Result<()> {
        let validate = |weight: &AffineQuantizedWeight, matrices, input, output, bits| {
            weight.validate(matrices, input, output, config.group_size, bits)
        };
        validate(&self.router, 1, config.hidden_size, config.expert_count, config.router_bits)?;
        for weight in [&self.routed_gate, &self.routed_up] {
            validate(
                weight,
                config.expert_count,
                config.hidden_size,
                config.routed_intermediate_size,
                config.expert_bits,
            )?;
        }
        validate(
            &self.routed_down,
            config.expert_count,
            config.routed_intermediate_size,
            config.hidden_size,
            config.expert_bits,
        )?;
        for weight in [&self.shared_gate, &self.shared_up] {
            validate(
                weight,
                1,
                config.hidden_size,
                config.shared_intermediate_size,
                config.expert_bits,
            )?;
        }
        validate(
            &self.shared_down,
            1,
            config.shared_intermediate_size,
            config.hidden_size,
            config.expert_bits,
        )?;
        validate(&self.shared_output_gate, 1, config.hidden_size, 1, config.router_bits)
    }
}
