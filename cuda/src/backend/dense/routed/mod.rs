//! Routed experts in place of a dense layer's gated MLP: the router scores
//! every expert, the unit router selects the top-k and weighs them as the
//! checkpoint declares (a softmax over the selected logits, probabilities
//! over every expert, sparsemixer bands, or a sigmoid on the one selected
//! expert's input), the selected experts' gated MLPs are summed with those
//! weights, and an optional shared expert adds its own.

mod ggml;
mod shared;
pub(super) mod weights;

pub(in crate::backend) use ggml::{GgmlExpertGeometry, GgmlRoutedExperts};
use mircuda::{DeviceBuffer, Stream, bf16};
pub use weights::RoutedExpertWeights;
use weights::RoutedExperts;
pub(in crate::backend) use weights::{ExpertBankLayout, GgmlExpertWeights};

use super::{DenseRoutedExperts, DenseSwiGluConfig};
use crate::{
    Bf16Projection, CudaBackend, DensePlanRequest, DenseRole, Error, ExecutionPhase, Result,
    backend::linear::SelectedDenseMoeBf16,
    kernels::{RouterInputScale, RouterUnitSpec, RouterUnitTopK, RouterUnitWeighting},
};

/// The routed feed-forward for a fixed number of rows.
pub(super) struct RoutedFeedForward {
    router: Bf16Projection,
    top_k: RouterUnitTopK,
    scores: DeviceBuffer<bf16>,
    selected: DeviceBuffer<u32>,
    routing: DeviceBuffer<bf16>,
    experts: Execution,
    input: ExpertInput,
    shared: Option<shared::SharedExpert>,
    stream: Stream,
}

/// What the selected experts read.
enum ExpertInput {
    /// The layer's normalized input, weighting the experts' outputs.
    Layer,
    /// Each row scaled by its one expert's weight (Llama 4), with unit
    /// weights on the outputs.
    Scaled {
        scale: RouterInputScale,
        rows: DeviceBuffer<bf16>,
    },
}

enum Execution {
    Dense {
        operation: Box<SelectedDenseMoeBf16>,
        intermediate: DeviceBuffer<bf16>,
    },
    Ggml(Box<GgmlRoutedExperts>),
}

impl RoutedFeedForward {
    pub(super) fn new(
        backend: &CudaBackend,
        config: DenseSwiGluConfig,
        rows: usize,
        weights: &RoutedExpertWeights,
    ) -> Result<Self> {
        let routed: DenseRoutedExperts = config
            .experts
            .ok_or(Error::InvalidExecutionPlan("dense layer config declares no routed experts"))?;
        let experts = routed.experts;
        let selections = rows
            .checked_mul(experts.top_k)
            .ok_or(Error::InvalidDecoderKernel("routed expert selection size overflow"))?;
        let execution = match &weights.experts {
            RoutedExperts::Dense(banks) => Execution::Dense {
                operation: Box::new(SelectedDenseMoeBf16::new(
                    backend,
                    rows,
                    experts.top_k,
                    banks,
                    config.activation.into(),
                )?),
                intermediate: backend.pool().allocate(
                    backend.stream(),
                    banks.intermediate_elements(rows, experts.top_k)?,
                )?,
            },
            RoutedExperts::Ggml(_) => {
                let geometry = GgmlExpertGeometry {
                    hidden: config.attention.hidden_size,
                    intermediate: config.intermediate_size,
                    experts,
                    activation: config.activation,
                };
                Execution::Ggml(Box::new(GgmlRoutedExperts::new(backend, geometry, rows)?))
            },
        };
        Ok(Self {
            router: backend.prepare_bf16_projection(DensePlanRequest {
                phase: if rows == 1 {
                    ExecutionPhase::Decode
                } else {
                    ExecutionPhase::Prefill
                },
                role: DenseRole::Router,
                tokens: rows,
                input_features: config.attention.hidden_size,
                output_features: experts.count,
            })?,
            top_k: RouterUnitTopK::compile_scored(
                backend.compiler(),
                RouterUnitSpec {
                    tokens: rows,
                    experts: experts.count,
                    top_k: experts.top_k,
                },
                routed.router(),
            )?,
            scores: backend.pool().allocate(
                backend.stream(),
                rows.checked_mul(experts.count)
                    .ok_or(Error::InvalidDecoderKernel("router score size overflow"))?,
            )?,
            selected: backend.pool().allocate(backend.stream(), selections)?,
            routing: backend.pool().allocate(backend.stream(), selections)?,
            experts: execution,
            input: match routed.weighting {
                // The semantic spec gives an input-scaled router one expert.
                RouterUnitWeighting::SigmoidInputScaled => {
                    let hidden = config.attention.hidden_size;
                    ExpertInput::Scaled {
                        scale: RouterInputScale::compile(backend.compiler(), hidden)?,
                        rows: backend
                            .pool()
                            .allocate(backend.stream(), super::scratch::elements(rows, hidden)?)?,
                    }
                },
                _ => ExpertInput::Layer,
            },
            shared: match (routed.shared_intermediate, &weights.shared) {
                (Some(intermediate), Some(shared)) => Some(shared::SharedExpert::new(
                    backend,
                    rows,
                    (config.attention.hidden_size, usize::try_from(intermediate.get())?),
                    config.activation,
                    shared,
                )?),
                (None, None) => None,
                _ => {
                    return Err(Error::InvalidExecutionPlan(
                        "routed layer config and weights disagree on a shared expert",
                    ));
                },
            },
            stream: backend.stream().clone(),
        })
    }

    /// Writes the routed MLP of the normalized `input` rows into `output`.
    pub(super) fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        weights: &RoutedExpertWeights,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.router.execute(input, &weights.router, &mut self.scores)?;
        self.top_k.execute(
            &self.stream,
            &self.scores,
            weights.correction_bias.as_ref(),
            &mut self.selected,
            &mut self.routing,
        )?;
        let expert_input = match &mut self.input {
            ExpertInput::Layer => input,
            ExpertInput::Scaled { scale, rows } => {
                scale.execute(&self.stream, input, &mut self.routing, rows)?;
                &*rows
            },
        };
        let routed = self.shared.as_mut().map_or(&mut *output, |shared| &mut shared.routed);
        match (&mut self.experts, &weights.experts) {
            (Execution::Dense { operation, intermediate }, RoutedExperts::Dense(banks)) => {
                operation.execute(
                    expert_input, &self.selected, &self.routing, banks, intermediate, routed,
                )
            },
            (Execution::Ggml(operation), RoutedExperts::Ggml(banks)) => {
                operation.execute(expert_input, &self.selected, &self.routing, banks, routed)
            },
            _ => Err(Error::InvalidExecutionPlan(
                "routed expert plan and late-bound expert storage differ",
            )),
        }?;
        match (&mut self.shared, &weights.shared) {
            (Some(shared), Some(shared_weights)) => shared.execute(input, shared_weights, output),
            (None, None) => Ok(()),
            _ => Err(Error::InvalidExecutionPlan(
                "routed plan and layer weights disagree on a shared expert",
            )),
        }
    }
}
