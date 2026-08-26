//! Shared routed feed-forward execution and projection fusion.

use models::weights::SharedRoutedFeedForwardBindings;

use super::{
    Array, Error, FusedExpertGateUp, FusedGateUp, ModelTensors, Result, Stream,
    binding::BoundLinear,
    fusion_planner::FusionPlanner,
    gate_up_tuning,
    lowering::FeedForwardLowering,
    route_tuning::{self, ExpertActivation, RoutingSpec},
};

mod load;
mod routed;
use load::{linear, projection_biases};
use routed::RoutedGateUp;

#[derive(Debug, Clone, Copy)]
pub struct SharedExpertMoeConfig {
    pub expert_count: usize,
    pub top_k: usize,
    pub intermediate: usize,
}

#[derive(Debug)]
pub struct SharedExpertMoe {
    config: SharedExpertMoeConfig,
    router: BoundLinear,
    routed_gate_up: RoutedGateUp,
    routed_down: BoundLinear,
    shared_gate: BoundLinear,
    shared_up: BoundLinear,
    fused_shared_gate_up: Option<FusedGateUp>,
    shared_down: BoundLinear,
    shared_output_gate: BoundLinear,
    fuse_shared_gate_up: bool,
}

impl SharedExpertMoeConfig {
    pub fn new(expert_count: usize, top_k: usize, intermediate: usize) -> Result<Self> {
        if expert_count == 0 || top_k == 0 || top_k > expert_count || intermediate == 0 {
            return Err(Error::InvalidModel(format!(
                "invalid shared-expert MoE dimensions: expert_count={expert_count}, top_k={top_k}"
            )));
        }
        Ok(Self { expert_count, top_k, intermediate })
    }
}

impl SharedExpertMoe {
    pub fn load(
        tensors: &ModelTensors,
        prefix: &str,
        config: SharedExpertMoeConfig,
        group_size: i32,
        stream: &Stream,
    ) -> Result<Self> {
        let routed_gate = linear(tensors, prefix, "switch_mlp.gate_proj", group_size)?;
        let routed_up = linear(tensors, prefix, "switch_mlp.up_proj", group_size)?;
        let shared_gate = linear(tensors, prefix, "shared_expert.gate_proj", group_size)?;
        let shared_up = linear(tensors, prefix, "shared_expert.up_proj", group_size)?;
        let fusion = FusionPlanner::new(stream).projections(
            FeedForwardLowering::SharedRouted,
            projection_biases(&shared_gate, &shared_up),
        );
        Ok(Self {
            config,
            router: linear(tensors, prefix, "gate", group_size)?,
            routed_gate_up: RoutedGateUp::Separate {
                gate: routed_gate,
                up: routed_up,
                fused: None,
            },
            routed_down: linear(tensors, prefix, "switch_mlp.down_proj", group_size)?,
            shared_gate,
            shared_up,
            fused_shared_gate_up: None,
            shared_down: linear(tensors, prefix, "shared_expert.down_proj", group_size)?,
            shared_output_gate: linear(tensors, prefix, "shared_expert_gate", group_size)?,
            fuse_shared_gate_up: fusion.gate_up,
        })
    }

    pub fn load_bindings(
        tensors: &ModelTensors,
        bindings: SharedRoutedFeedForwardBindings<'_>,
        config: SharedExpertMoeConfig,
        lowering: FeedForwardLowering,
        stream: &Stream,
    ) -> Result<Self> {
        let (routed_gate_up, routed_down) = RoutedGateUp::load(tensors, bindings.routed, stream)?;
        let shared_gate = BoundLinear::load(tensors, bindings.shared_gate, stream)?;
        let shared_up = BoundLinear::load(tensors, bindings.shared_up, stream)?;
        let fusion = FusionPlanner::new(stream)
            .projections(lowering, projection_biases(&shared_gate, &shared_up));
        Ok(Self {
            config,
            router: BoundLinear::load(tensors, bindings.router, stream)?,
            routed_gate_up,
            routed_down,
            shared_gate,
            shared_up,
            fused_shared_gate_up: None,
            shared_down: BoundLinear::load(tensors, bindings.shared_down, stream)?,
            shared_output_gate: BoundLinear::load(tensors, bindings.shared_output_gate, stream)?,
            fuse_shared_gate_up: fusion.gate_up,
        })
    }

    pub fn forward(&self, input: &Array, stream: &Stream) -> Result<Array> {
        let scores = self.router.forward(input, stream)?;
        let routing = scores.router_top_k_unit(i32::try_from(self.config.top_k)?, stream)?;
        let routed = self.routed(input, &routing.indices, &routing.weights, stream)?;
        routed.add(&self.shared(input, stream)?, stream)
    }

    pub(crate) fn enable_routed_gate_up(&mut self, stream: &Stream) -> Result<bool> {
        if self.routed_gate_up.is_fused() {
            return Ok(true);
        }
        let routed = self.routed_gate_up.enable(stream)?;
        self.fused_shared_gate_up = self
            .fuse_shared_gate_up
            .then(|| self.shared_gate.fuse_gate_up(&self.shared_up, stream))
            .transpose()?
            .flatten();
        self.fused_shared_gate_up.as_ref().map_or(Ok(()), |fused| fused.warm(stream))?;
        Ok(routed)
    }

    pub(crate) fn fused_routed_gate_up_bytes(&self) -> Result<Option<usize>> {
        let routed = self.routed_gate_up.fused_bytes()?;
        if !self.fuse_shared_gate_up {
            return Ok(routed);
        }
        let shared = self.shared_gate.fused_gate_up_bytes(&self.shared_up)?;
        match (routed, shared) {
            (Some(routed), Some(shared)) => {
                routed.checked_add(shared).map(Some).ok_or(Error::ShapeOverflow)
            },
            _ => Ok(None),
        }
    }

    pub(crate) const fn has_fused_routed_gate_up(&self) -> bool {
        self.routed_gate_up.is_fused()
    }

    fn routed(
        &self,
        input: &Array,
        indices: &Array,
        weights: &Array,
        stream: &Stream,
    ) -> Result<Array> {
        let (group_size, bits) = self.routed_gate_up.tuning_format();
        let spec = RoutingSpec {
            experts: self.config.expert_count,
            intermediate: self.config.intermediate,
            group_size,
            bits,
            activation: ExpertActivation::Silu,
            fused_unsorted: self.routed_gate_up.is_fused(),
        };
        route_tuning::forward(
            spec,
            input,
            indices,
            stream,
            (
                |indices| {
                    let sorted = input.sort_expert_inputs(indices, stream)?;
                    let output =
                        self.routed_mlp(&sorted.input, &sorted.indices, true, false, stream)?;
                    sorted.restore(&output, stream)?.weighted_sum(weights, -2, stream)
                },
                |indices| {
                    let sorted = input.sort_expert_inputs(indices, stream)?;
                    let output =
                        self.routed_mlp(&sorted.input, &sorted.indices, true, false, stream)?;
                    sorted.restore_weighted(&output, weights, stream)
                },
                |indices| {
                    let grouped =
                        input.group_expert_inputs(indices, self.config.expert_count, stream)?;
                    let output =
                        self.routed_mlp(&grouped.input, &grouped.indices, true, false, stream)?;
                    grouped.restore_weighted(&output, weights, stream)
                },
                |indices| {
                    let input = input.expand_dims(&[-2, -3], stream)?;
                    self.routed_mlp(&input, indices, false, false, stream)?
                        .squeeze_axis(-2, stream)?
                        .weighted_sum(weights, -2, stream)
                },
                |indices| {
                    let input = input.expand_dims(&[-2, -3], stream)?;
                    self.routed_mlp(&input, indices, false, true, stream)?
                        .squeeze_axis(-2, stream)?
                        .weighted_sum(weights, -2, stream)
                },
            ),
        )
    }

    fn routed_mlp(
        &self,
        input: &Array,
        indices: &Array,
        sorted: bool,
        native_output: bool,
        stream: &Stream,
    ) -> Result<Array> {
        let (gate, up) = if native_output {
            self.routed_gate_up.gather_native(input, indices, stream)?
        } else {
            self.routed_gate_up.gather(input, indices, sorted, stream)?
        };
        let activated = gate.silu_mul(&up, stream)?;
        if native_output {
            self.routed_down.gather_native(&activated, indices, false, stream)
        } else {
            self.routed_down.gather(&activated, indices, sorted, stream)
        }
    }

    fn shared(&self, input: &Array, stream: &Stream) -> Result<Array> {
        let fused = self.fused_shared_gate_up.as_ref();
        let (gate, up) = if gate_up_tuning::is_single_token(input)? {
            gate_up_tuning::forward(&self.shared_gate, &self.shared_up, fused, input, stream)?
        } else {
            fused.map_or_else(
                || {
                    Ok((
                        self.shared_gate.forward(input, stream)?,
                        self.shared_up.forward(input, stream)?,
                    ))
                },
                |fused| fused.forward_pair(input, stream),
            )?
        };
        let output = self.shared_down.forward(&gate.silu_mul(&up, stream)?, stream)?;
        self.shared_output_gate.forward(input, stream)?.sigmoid_mul(&output, stream)
    }
}

#[cfg(test)]
mod tests;
