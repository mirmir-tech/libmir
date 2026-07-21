use super::{
    AttentionProjectionRole, ExpertProjectionRole, LayerTensorRole, LogicalTensorRole,
    TensorBinding, WeightBindingPlan,
};
use crate::error::{ModelsError, Result};

#[derive(Debug, Clone, Copy)]
pub struct DecoderBoundaryBindings<'a> {
    pub embedding: &'a TensorBinding,
    pub final_norm: &'a TensorBinding,
    pub output: &'a TensorBinding,
}

#[derive(Debug, Clone, Copy)]
pub struct RoutedDecoderLayerBindings<'a> {
    pub input_norm: &'a TensorBinding,
    pub query: &'a TensorBinding,
    pub key: &'a TensorBinding,
    pub value: &'a TensorBinding,
    pub attention_output: &'a TensorBinding,
    pub attention_sinks: &'a TensorBinding,
    pub post_attention_norm: &'a TensorBinding,
    pub router: &'a TensorBinding,
    pub experts: RoutedExpertBindings<'a>,
}

#[derive(Debug, Clone, Copy)]
pub enum RoutedExpertBindings<'a> {
    InterleavedGateUp {
        gate_up: &'a TensorBinding,
        down: &'a TensorBinding,
    },
    SeparateGateUp {
        gate: &'a TensorBinding,
        up: &'a TensorBinding,
        down: &'a TensorBinding,
    },
}

impl WeightBindingPlan {
    pub fn decoder_boundary(&self) -> Result<DecoderBoundaryBindings<'_>> {
        Ok(DecoderBoundaryBindings {
            embedding: required(self, &LogicalTensorRole::Embedding)?,
            final_norm: required(self, &LogicalTensorRole::FinalNorm)?,
            output: required(self, &LogicalTensorRole::Output)?,
        })
    }

    pub fn decoder_boundary_with_tied_output(
        &self,
        tied: bool,
    ) -> Result<DecoderBoundaryBindings<'_>> {
        let embedding = required(self, &LogicalTensorRole::Embedding)?;
        Ok(DecoderBoundaryBindings {
            embedding,
            final_norm: required(self, &LogicalTensorRole::FinalNorm)?,
            output: if tied {
                embedding
            } else {
                required(self, &LogicalTensorRole::Output)?
            },
        })
    }

    pub fn routed_decoder_layer(&self, index: usize) -> Result<RoutedDecoderLayerBindings<'_>> {
        Ok(RoutedDecoderLayerBindings {
            input_norm: layer(self, index, LayerTensorRole::InputNorm)?,
            query: attention(self, index, AttentionProjectionRole::Query)?,
            key: attention(self, index, AttentionProjectionRole::Key)?,
            value: attention(self, index, AttentionProjectionRole::Value)?,
            attention_output: attention(self, index, AttentionProjectionRole::Output)?,
            attention_sinks: layer(self, index, LayerTensorRole::AttentionSinks)?,
            post_attention_norm: layer(self, index, LayerTensorRole::PostAttentionNorm)?,
            router: layer(self, index, LayerTensorRole::Router)?,
            experts: experts(self, index)?,
        })
    }
}

impl<'a> DecoderBoundaryBindings<'a> {
    #[must_use]
    pub fn physical_sources(self) -> Vec<&'a str> {
        sources([self.embedding, self.final_norm, self.output])
    }
}

impl<'a> RoutedDecoderLayerBindings<'a> {
    #[must_use]
    pub fn physical_sources(self) -> Vec<&'a str> {
        let mut bindings = vec![
            self.input_norm,
            self.query,
            self.key,
            self.value,
            self.attention_output,
            self.attention_sinks,
            self.post_attention_norm,
            self.router,
        ];
        match self.experts {
            RoutedExpertBindings::InterleavedGateUp { gate_up, down } => {
                bindings.extend([gate_up, down]);
            },
            RoutedExpertBindings::SeparateGateUp { gate, up, down } => {
                bindings.extend([gate, up, down]);
            },
        }
        sources(bindings)
    }
}

fn experts(plan: &WeightBindingPlan, index: usize) -> Result<RoutedExpertBindings<'_>> {
    let down = expert(plan, index, ExpertProjectionRole::Down)?;
    if let Some(gate_up) = optional_expert(plan, index, ExpertProjectionRole::GateUp) {
        return Ok(RoutedExpertBindings::InterleavedGateUp { gate_up, down });
    }
    Ok(RoutedExpertBindings::SeparateGateUp {
        gate: expert(plan, index, ExpertProjectionRole::Gate)?,
        up: expert(plan, index, ExpertProjectionRole::Up)?,
        down,
    })
}

fn attention(
    plan: &WeightBindingPlan,
    index: usize,
    projection: AttentionProjectionRole,
) -> Result<&TensorBinding> {
    layer(plan, index, LayerTensorRole::AttentionProjection { projection })
}

fn expert(
    plan: &WeightBindingPlan,
    index: usize,
    projection: ExpertProjectionRole,
) -> Result<&TensorBinding> {
    layer(plan, index, expert_role(projection))
}

fn optional_expert(
    plan: &WeightBindingPlan,
    index: usize,
    projection: ExpertProjectionRole,
) -> Option<&TensorBinding> {
    plan.binding(&LogicalTensorRole::Layer { index, tensor: expert_role(projection) })
}

const fn expert_role(projection: ExpertProjectionRole) -> LayerTensorRole {
    LayerTensorRole::ExpertProjection { expert: None, projection }
}

fn layer(
    plan: &WeightBindingPlan,
    index: usize,
    tensor: LayerTensorRole,
) -> Result<&TensorBinding> {
    required(plan, &LogicalTensorRole::Layer { index, tensor })
}

fn required<'a>(
    plan: &'a WeightBindingPlan,
    role: &LogicalTensorRole,
) -> Result<&'a TensorBinding> {
    plan.binding(role).ok_or_else(|| {
        ModelsError::InvalidConfig(format!("logical tensor role is unbound: {role:?}"))
    })
}

fn sources<'a>(bindings: impl IntoIterator<Item = &'a TensorBinding>) -> Vec<&'a str> {
    bindings.into_iter().flat_map(TensorBinding::physical_sources).collect()
}
