//! Routed-expert block: sigmoid scores biased by a checkpoint correction
//! select the top-k experts, whose renormalized raw sigmoids weight the
//! ungated `relu²` expert outputs. Experts may run in a projected latent
//! space; an optional shared expert always runs at model width.

use mirtal::DType;
use models::{
    layout::LatentRoutedConfig,
    weights::{ExpertProjectionRole, LatentRoutedBindings, RoutedExpertBindings},
};

use crate::engine::{
    Array, Error, ModelTensors, NormWeight, Result, RouterOutput, Stream, binding::BoundLinear,
};

const SCORE_EPSILON: f32 = 1e-20;

#[derive(Debug)]
pub(super) struct RoutedFeedForward {
    pub(super) norm: NormWeight,
    router: BoundLinear,
    correction_bias: Array,
    latent: Option<(BoundLinear, BoundLinear)>,
    up: BoundLinear,
    down: BoundLinear,
    shared: Option<(BoundLinear, BoundLinear)>,
    top_k: usize,
    routed_scale: f32,
    epsilon: f32,
}

impl RoutedFeedForward {
    pub(super) fn load(
        tensors: &ModelTensors,
        bindings: LatentRoutedBindings<'_>,
        config: &LatentRoutedConfig,
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
            correction_bias: tensors.get(&bindings.correction_bias.source)?,
            latent,
            up,
            down,
            shared,
            top_k: config.top_k,
            routed_scale: config.routed_scale.to_string().parse()?,
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
        let graph = stream.native().graph();
        let logits = self.router.forward(input, stream)?;
        let scores = graph.sigmoid(&graph.astype(logits.native(), DType::Float32)?)?;
        let biased = graph.add(&scores, self.correction_bias.native())?;
        let shape = biased.shape()?;
        let rank = shape.dimensions().len();
        let top_k = self.top_k;
        let kth = i32::try_from(top_k - 1)?;
        let order = graph.argpartition(&graph.negative(&biased)?, kth, -1)?;
        let mut stop = shape.dimensions().to_vec();
        stop[rank - 1] = top_k;
        let indices = graph.slice(&order, &vec![0; rank], &stop)?;
        let selected = graph.take_along_axis(&scores, &indices, -1)?;
        let weights = if top_k > 1 {
            let sum = graph.reduce_sum(&selected, -1, true)?;
            graph.divide(&selected, &graph.add_scalar(&sum, SCORE_EPSILON)?)?
        } else {
            selected
        };
        let weights = graph.multiply_scalar(&weights, self.routed_scale)?;
        Ok(RouterOutput {
            indices: Array::from_native(graph.astype(&indices, DType::Uint32)?)?,
            weights: Array::from_native(weights)?,
        })
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
