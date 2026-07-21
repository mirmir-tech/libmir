//! Shared routed feed-forward execution and projection fusion.

use models::weights::SharedRoutedFeedForwardBindings;

use super::{
    Array, Error, FusedExpertGateUp, FusedGateUp, ModelTensors, QuantizedLinear, Result, Stream,
    binding::affine_linear,
    fusion_planner::{FusionPlanner, ProjectionBiases},
    lowering::FeedForwardLowering,
};

#[derive(Debug, Clone, Copy)]
pub struct SharedExpertMoeConfig {
    pub expert_count: usize,
    pub top_k: usize,
}

#[derive(Debug)]
pub struct SharedExpertMoe {
    config: SharedExpertMoeConfig,
    router: QuantizedLinear,
    routed_gate: QuantizedLinear,
    routed_up: QuantizedLinear,
    routed_down: QuantizedLinear,
    fused_routed_gate_up: Option<FusedExpertGateUp>,
    shared_gate: QuantizedLinear,
    shared_up: QuantizedLinear,
    fused_shared_gate_up: Option<FusedGateUp>,
    shared_down: QuantizedLinear,
    shared_output_gate: QuantizedLinear,
    fuse_shared_gate_up: bool,
}

impl SharedExpertMoeConfig {
    pub fn new(expert_count: usize, top_k: usize) -> Result<Self> {
        if expert_count == 0 || top_k == 0 || top_k > expert_count {
            return Err(Error::InvalidModel(format!(
                "invalid shared-expert MoE dimensions: expert_count={expert_count}, top_k={top_k}"
            )));
        }
        Ok(Self { expert_count, top_k })
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
            routed_gate,
            routed_up,
            routed_down: linear(tensors, prefix, "switch_mlp.down_proj", group_size)?,
            fused_routed_gate_up: None,
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
        let routed_gate = affine_linear(tensors, bindings.routed_gate)?;
        let routed_up = affine_linear(tensors, bindings.routed_up)?;
        let shared_gate = affine_linear(tensors, bindings.shared_gate)?;
        let shared_up = affine_linear(tensors, bindings.shared_up)?;
        let fusion = FusionPlanner::new(stream)
            .projections(lowering, projection_biases(&shared_gate, &shared_up));
        Ok(Self {
            config,
            router: affine_linear(tensors, bindings.router)?,
            routed_gate,
            routed_up,
            routed_down: affine_linear(tensors, bindings.routed_down)?,
            fused_routed_gate_up: None,
            shared_gate,
            shared_up,
            fused_shared_gate_up: None,
            shared_down: affine_linear(tensors, bindings.shared_down)?,
            shared_output_gate: affine_linear(tensors, bindings.shared_output_gate)?,
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
        if self.fused_routed_gate_up.is_some() {
            return Ok(true);
        }
        self.fused_routed_gate_up =
            self.routed_gate.fuse_expert_gate_up(&self.routed_up, stream)?;
        self.fused_shared_gate_up = self
            .fuse_shared_gate_up
            .then(|| self.shared_gate.fuse_gate_up(&self.shared_up, stream))
            .transpose()?
            .flatten();
        self.fused_routed_gate_up.as_ref().map_or(Ok(()), FusedExpertGateUp::warm)?;
        self.fused_shared_gate_up.as_ref().map_or(Ok(()), FusedGateUp::warm)?;
        Ok(self.fused_routed_gate_up.is_some())
    }

    pub(crate) fn fused_routed_gate_up_bytes(&self) -> Result<Option<usize>> {
        let routed = self.routed_gate.fused_expert_gate_up_bytes(&self.routed_up)?;
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
        self.fused_routed_gate_up.is_some()
    }

    fn routed(
        &self,
        input: &Array,
        indices: &Array,
        weights: &Array,
        stream: &Stream,
    ) -> Result<Array> {
        if should_sort(indices)? {
            let sorted = input.sort_expert_inputs(indices, stream)?;
            let output = self.routed_mlp(&sorted.input, &sorted.indices, true, stream)?;
            return sorted.restore(&output, stream)?.weighted_sum(weights, -2, stream);
        }
        let input = input.expand_dims(&[-2, -3], stream)?;
        self.routed_mlp(&input, indices, false, stream)?
            .squeeze_axis(-2, stream)?
            .weighted_sum(weights, -2, stream)
    }

    fn routed_mlp(
        &self,
        input: &Array,
        indices: &Array,
        sorted: bool,
        stream: &Stream,
    ) -> Result<Array> {
        let fused = (!sorted).then_some(self.fused_routed_gate_up.as_ref()).flatten();
        let (gate, up) = fused.map_or_else(
            || {
                Ok((
                    self.routed_gate.gather(input, indices, sorted, stream)?,
                    self.routed_up.gather(input, indices, sorted, stream)?,
                ))
            },
            |fused| fused.forward(input, indices, stream),
        )?;
        let activated = gate.silu_mul(&up, stream)?;
        self.routed_down.gather(&activated, indices, sorted, stream)
    }

    fn shared(&self, input: &Array, stream: &Stream) -> Result<Array> {
        let (gate, up) = self.fused_shared_gate_up.as_ref().map_or_else(
            || {
                Ok((
                    self.shared_gate.forward(input, stream)?,
                    self.shared_up.forward(input, stream)?,
                ))
            },
            |fused| fused.forward_pair(input, stream),
        )?;
        let output = self.shared_down.forward(&gate.silu_mul(&up, stream)?, stream)?;
        self.shared_output_gate.forward(input, stream)?.sigmoid_mul(&output, stream)
    }
}

fn should_sort(indices: &Array) -> Result<bool> {
    indices
        .shape()?
        .into_iter()
        .try_fold(1_usize, |count, dimension| {
            count.checked_mul(usize::try_from(dimension)?).ok_or(Error::ShapeOverflow)
        })
        .map(|count| count >= 64)
}

fn projection_biases(gate: &QuantizedLinear, up: &QuantizedLinear) -> ProjectionBiases {
    ProjectionBiases::new([false, false], None, [gate.has_bias(), up.has_bias()])
}

fn linear(
    tensors: &ModelTensors,
    prefix: &str,
    name: &str,
    group_size: i32,
) -> Result<QuantizedLinear> {
    QuantizedLinear::load(tensors, &format!("{prefix}.{name}"), group_size)
}

#[cfg(test)]
mod tests;
