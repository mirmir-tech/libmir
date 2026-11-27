//! Expert selection of a shared-expert mixture: the renormalized softmax top-k
//! of Qwen-style routers, or a configured router (`DeepSeek`): softmax
//! without renormalization or sigmoid with a selection-only correction bias,
//! optionally limited to the best expert groups, scaled by
//! `routed_scaling_factor`; or Phi-3.5-MoE's sparsemixer.

mod sparse_mixer;

use mirtal::{DType, Graph};
use models::semantic::ExpertGroupLimit;

use crate::engine::{Array, Error, Result, RouterOutput, Stream, binding::BoundLinear};

/// Keeps sigmoid renormalization finite when every selected score is zero.
const SCORE_EPSILON: f32 = 1e-20;
/// Subtracted from the selection scores of dropped expert groups.
const DROPPED: f32 = 1e9;

#[derive(Debug)]
pub enum ExpertRouter {
    /// Softmax top-k, renormalized (`route_unit`).
    Unit,
    Scored(ScoredRouter),
    /// The top logit's sigmoid scales the selected expert's input
    /// (Llama 4); routing weights are then one.
    SigmoidInputScaled,
    /// Phi-3.5-MoE's sparsemixer with its jitter.
    SparseMixer {
        jitter: f32,
    },
}

/// How a scored router turns logits into selection scores.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoredRouting {
    /// Softmax over the experts, renormalized over the selection or not.
    Softmax { renormalize: bool },
    /// Sigmoid scores, renormalized over the selection, chosen with a
    /// selection-only correction bias.
    SigmoidWithCorrectionBias,
}

#[derive(Debug)]
pub struct ScoredRouter {
    pub routing: ScoredRouting,
    /// `[experts]` F32 bias added to sigmoid scores for selection only.
    pub correction_bias: Option<Array>,
    pub group_limit: Option<ExpertGroupLimit>,
    pub output_scale: f32,
}

impl ExpertRouter {
    pub(super) fn route(
        &self,
        router: &BoundLinear,
        input: &Array,
        top_k: usize,
        stream: &Stream,
    ) -> Result<RouterOutput> {
        match self {
            Self::Unit => router.route_unit(input, i32::try_from(top_k)?, stream),
            Self::SigmoidInputScaled => {
                let logits = router.forward(input, stream)?;
                let graph = stream.native().graph();
                let shape = logits.shape()?;
                let rank = shape.len();
                let order = graph.argpartition(
                    &graph.negative(logits.native())?,
                    i32::try_from(top_k - 1)?,
                    -1,
                )?;
                let mut stop = shape
                    .iter()
                    .map(|&dim| usize::try_from(dim))
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                stop[rank - 1] = top_k;
                let indices = graph.slice(&order, &vec![0; rank], &stop)?;
                let selected = graph.take_along_axis(logits.native(), &indices, -1)?;
                let scores = graph.sigmoid(&graph.astype(&selected, DType::Float32)?)?;
                Ok(RouterOutput {
                    indices: Array::from_native(graph.astype(&indices, DType::Uint32)?)?,
                    weights: Array::from_native(scores)?.astype_like(&logits, stream)?,
                })
            },
            Self::SparseMixer { jitter } => {
                let logits = router.forward(input, stream)?;
                let routing =
                    sparse_mixer::route(&logits, top_k, *jitter, stream.native().graph())?;
                // Expert reductions take weights in the activations' dtype.
                Ok(RouterOutput {
                    weights: routing.weights.astype_like(input, stream)?,
                    indices: routing.indices,
                })
            },
            Self::Scored(scored) => {
                let logits = router.forward(input, stream)?;
                let output = scored.route(&logits, top_k, stream)?;
                // Expert reductions take weights in the activations' dtype.
                Ok(RouterOutput {
                    weights: output.weights.astype_like(&logits, stream)?,
                    indices: output.indices,
                })
            },
        }
    }
}

impl ScoredRouter {
    pub(in crate::engine) fn route(
        &self,
        logits: &Array,
        top_k: usize,
        stream: &Stream,
    ) -> Result<RouterOutput> {
        let graph = stream.native().graph();
        let logits = graph.astype(logits.native(), DType::Float32)?;
        let (scores, renormalize) = match self.routing {
            ScoredRouting::Softmax { renormalize } => {
                (graph.softmax(&logits, -1, true)?, renormalize)
            },
            ScoredRouting::SigmoidWithCorrectionBias => (graph.sigmoid(&logits)?, true),
        };
        let selection = match &self.correction_bias {
            Some(bias) => graph.add(&scores, bias.native())?,
            None => scores.clone(),
        };
        let selection = match self.group_limit.filter(|limit| limit.groups > 1) {
            Some(limit) => self.limit_groups(graph, &selection, limit)?,
            None => selection,
        };
        let shape = selection.shape()?;
        let rank = shape.dimensions().len();
        let order =
            graph.argpartition(&graph.negative(&selection)?, i32::try_from(top_k - 1)?, -1)?;
        let mut stop = shape.dimensions().to_vec();
        stop[rank - 1] = top_k;
        let indices = graph.slice(&order, &vec![0; rank], &stop)?;
        let selected = graph.take_along_axis(&scores, &indices, -1)?;
        // transformers renormalizes whenever the router does, top-1 included
        // (DeepSeek-V3 and GLM-4.5 `noaux_tc`, Qwen3-MoE `norm_topk_prob`);
        // only DeepSeek-V2 guards on `top_k > 1`, and it never renormalizes.
        let weights = if renormalize {
            let sum = graph.reduce_sum(&selected, -1, true)?;
            graph.divide(&selected, &graph.add_scalar(&sum, SCORE_EPSILON)?)?
        } else {
            selected
        };
        Ok(RouterOutput {
            indices: Array::from_native(graph.astype(&indices, DType::Uint32)?)?,
            weights: Array::from_native(graph.multiply_scalar(&weights, self.output_scale)?)?,
        })
    }

    /// Selection scores outside the `limit.selected` best groups lowered by
    /// [`DROPPED`]. Softmax routers rank a group by its best score, sigmoid
    /// routers by the sum of its two best biased scores.
    fn limit_groups(
        &self,
        graph: Graph<'_>,
        selection: &mirtal::Array,
        limit: ExpertGroupLimit,
    ) -> Result<mirtal::Array> {
        let shape = selection.shape()?.dimensions().to_vec();
        let [leading @ .., experts] = shape.as_slice() else {
            return Err(Error::InvalidModel("router scores have no expert axis".into()));
        };
        if limit.selected == 0 || !experts.is_multiple_of(limit.groups) {
            return Err(Error::InvalidModel("expert groups do not divide the experts".into()));
        }
        let mut grouped_shape = leading.to_vec();
        grouped_shape.extend([limit.groups, experts / limit.groups]);
        let grouped = graph.reshape(selection, &mirtal::Shape::new(grouped_shape.clone())?)?;
        let group_scores = match self.routing {
            ScoredRouting::Softmax { .. } => graph.reduce_max(&grouped, -1, false)?,
            ScoredRouting::SigmoidWithCorrectionBias => {
                let order = graph.argpartition(&graph.negative(&grouped)?, 1, -1)?;
                let mut stop = grouped_shape.clone();
                if let Some(last) = stop.last_mut() {
                    *last = 2;
                }
                let best = graph.slice(&order, &vec![0; stop.len()], &stop)?;
                graph.reduce_sum(&graph.take_along_axis(&grouped, &best, -1)?, -1, false)?
            },
        };
        // A group's rank is the inverse permutation of the descending order;
        // `selected - rank`, clipped to [0, 1], keeps the best groups.
        let order = graph.argsort(&graph.negative(&group_scores)?, -1)?;
        let ranks = graph.astype(&graph.argsort(&order, -1)?, DType::Float32)?;
        let scalar = |value: f32| graph.full(&mirtal::Shape::new([])?, value, DType::Float32);
        let keep = graph.clip(
            &graph
                .add_scalar(&graph.negative(&ranks)?, f32::from(u16::try_from(limit.selected)?))?,
            &scalar(0.0)?,
            &scalar(1.0)?,
        )?;
        let keep = graph.expand_dims(&keep, &[-1])?;
        let dropped = graph.multiply_scalar(&graph.add_scalar(&keep, -1.0)?, DROPPED)?;
        let lowered = graph.add(&grouped, &dropped)?;
        Ok(graph.reshape(&lowered, &mirtal::Shape::new(shape.clone())?)?)
    }
}
