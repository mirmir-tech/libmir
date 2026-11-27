//! Routed-expert block: sigmoid scores biased by a checkpoint correction
//! select the top-k experts, whose renormalized raw sigmoids weight the
//! ungated `relu²` expert outputs. Experts may run in a projected latent
//! space; an optional shared expert always runs at model width.

use models::{
    semantic::RoutedExpertsSpec,
    weights::{ExpertProjectionRole, LatentRoutedBindings, RoutedExpertBindings},
};

use crate::engine::{
    Array, Error, ModelTensors, NormWeight, Result, RouterOutput, ScoredRouter, ScoredRouting,
    Stream, binding::BoundLinear,
};

#[derive(Debug)]
pub(in crate::engine::state_space_hybrid) struct RoutedFeedForward {
    pub(in crate::engine::state_space_hybrid) norm: NormWeight,
    router: BoundLinear,
    /// The shared sigmoid router with a selection-only correction bias.
    scored: ScoredRouter,
    latent: Option<(BoundLinear, BoundLinear)>,
    up: BoundLinear,
    down: BoundLinear,
    shared: Option<(BoundLinear, BoundLinear)>,
    top_k: usize,
    epsilon: f32,
}

impl RoutedFeedForward {
    pub(super) fn load(
        tensors: &ModelTensors,
        bindings: LatentRoutedBindings<'_>,
        spec: &RoutedExpertsSpec,
        epsilon: f32,
        stream: &Stream,
    ) -> Result<Self> {
        let latent = bindings
            .latent
            .map(|latent| {
                Ok::<_, Error>((
                    BoundLinear::load(tensors, latent.input, stream)?,
                    BoundLinear::load(tensors, latent.output, stream)?,
                ))
            })
            .transpose()?;
        let (up, down) = load_experts(tensors, bindings.routed, stream)?;
        let shared = bindings
            .shared
            .map(|shared| {
                Ok::<_, Error>((
                    BoundLinear::load(tensors, shared.up, stream)?,
                    BoundLinear::load(tensors, shared.down, stream)?,
                ))
            })
            .transpose()?;
        Ok(Self {
            norm: NormWeight::load_name(tensors, &bindings.norm.source)?,
            router: BoundLinear::load(tensors, bindings.router, stream)?,
            scored: ScoredRouter {
                routing: ScoredRouting::SigmoidWithCorrectionBias,
                correction_bias: Some(tensors.get(&bindings.correction_bias.source)?),
                group_limit: None,
                output_scale: spec.output_scale.to_string().parse()?,
            },
            latent,
            up,
            down,
            shared,
            top_k: spec.top_k,
            epsilon,
        })
    }

    pub(super) const fn epsilon(&self) -> f32 {
        self.epsilon
    }

    /// Runs the block over a normalized `[rows, sequence, hidden]` input.
    pub(super) fn forward(&self, input: &Array, stream: &Stream) -> Result<Array> {
        let RouterOutput { indices, weights } = self.route(input, stream)?;
        let projected;
        let expert_input = if let Some((latent_in, _)) = &self.latent {
            projected = latent_in.forward(input, stream)?;
            &projected
        } else {
            input
        };
        let routed = self.experts(expert_input, &indices, &weights, stream)?;
        let routed = match &self.latent {
            Some((_, latent_out)) => latent_out.forward(&routed, stream)?,
            None => routed,
        };
        match &self.shared {
            Some((up, down)) => {
                let shared =
                    down.forward(&up.forward(input, stream)?.relu_squared(stream)?, stream)?;
                routed.add(&shared, stream)
            },
            None => Ok(routed),
        }
    }

    fn route(&self, input: &Array, stream: &Stream) -> Result<RouterOutput> {
        let logits = self.router.forward(input, stream)?;
        self.scored.route(&logits, self.top_k, stream)
    }

    fn experts(
        &self,
        input: &Array,
        indices: &Array,
        weights: &Array,
        stream: &Stream,
    ) -> Result<Array> {
        if indices.native().len() >= 64 {
            let sorted = input.sort_expert_inputs(indices, stream)?;
            let output = self.mlp(&sorted.input, &sorted.indices, true, stream)?;
            return sorted.restore(&output, stream)?.weighted_sum(weights, -2, stream);
        }
        let input = input.expand_dims(&[-2, -3], stream)?;
        self.mlp(&input, indices, false, stream)?
            .squeeze_axis(-2, stream)?
            .weighted_sum(weights, -2, stream)
    }

    fn mlp(&self, input: &Array, indices: &Array, sorted: bool, stream: &Stream) -> Result<Array> {
        let up = self.up.gather(input, indices, sorted, stream)?;
        self.down.gather(&up.relu_squared(stream)?, indices, sorted, stream)
    }
}

fn load_experts(
    tensors: &ModelTensors,
    bindings: RoutedExpertBindings<'_>,
    stream: &Stream,
) -> Result<(BoundLinear, BoundLinear)> {
    match bindings {
        RoutedExpertBindings::UngatedStacked { up, down } => Ok((
            BoundLinear::load(tensors, up, stream)?,
            BoundLinear::load(tensors, down, stream)?,
        )),
        RoutedExpertBindings::Individual { .. } => Ok((
            BoundLinear::load_individual_bank(
                tensors,
                &bindings.individual(ExpertProjectionRole::Up),
                stream,
            )?,
            BoundLinear::load_individual_bank(
                tensors,
                &bindings.individual(ExpertProjectionRole::Down),
                stream,
            )?,
        )),
        _ => Err(Error::InvalidModel(
            "routed experts require stacked or individual up/down tensors without gates".into(),
        )),
    }
}
