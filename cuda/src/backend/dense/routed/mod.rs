//! Routed experts in place of a dense layer's gated MLP, without a shared
//! expert: the router scores every expert, the top-k are renormalized with a
//! softmax over their logits, and the selected experts' gated MLPs are summed
//! with those weights.

mod ggml;
pub(super) mod weights;

pub(in crate::backend) use ggml::{GgmlExpertGeometry, GgmlRoutedExperts};
pub(in crate::backend) use weights::GgmlExpertWeights;

use mircuda::{DeviceBuffer, Stream, bf16};
pub use weights::RoutedExpertWeights;
use weights::RoutedExperts;

use super::DenseSwiGluConfig;
use crate::{
    Bf16Projection, CudaBackend, DensePlanRequest, DenseRole, Error, ExecutionPhase, Result,
    backend::linear::SelectedDenseMoeBf16,
    kernels::{RouterUnitSpec, RouterUnitTopK},
};

/// The routed feed-forward for a fixed number of rows.
pub(super) struct RoutedFeedForward {
    router: Bf16Projection,
    top_k: RouterUnitTopK,
    scores: DeviceBuffer<bf16>,
    selected: DeviceBuffer<u32>,
    routing: DeviceBuffer<bf16>,
    experts: Execution,
    stream: Stream,
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
        let experts = config
            .experts
            .ok_or(Error::InvalidExecutionPlan("dense layer config declares no routed experts"))?;
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
            top_k: RouterUnitTopK::compile(
                backend.compiler(),
                RouterUnitSpec {
                    tokens: rows,
                    experts: experts.count,
                    top_k: experts.top_k,
                },
            )?,
            scores: backend.pool().allocate(
                backend.stream(),
                rows.checked_mul(experts.count)
                    .ok_or(Error::InvalidDecoderKernel("router score size overflow"))?,
            )?,
            selected: backend.pool().allocate(backend.stream(), selections)?,
            routing: backend.pool().allocate(backend.stream(), selections)?,
            experts: execution,
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
        self.top_k
            .execute(&self.stream, &self.scores, &mut self.selected, &mut self.routing)?;
        match (&mut self.experts, &weights.experts) {
            (Execution::Dense { operation, intermediate }, RoutedExperts::Dense(banks)) => {
                operation.execute(input, &self.selected, &self.routing, banks, intermediate, output)
            },
            (Execution::Ggml(operation), RoutedExperts::Ggml(banks)) => {
                operation.execute(input, &self.selected, &self.routing, banks, output)
            },
            _ => Err(Error::InvalidExecutionPlan(
                "routed expert plan and late-bound expert storage differ",
            )),
        }
    }
}
