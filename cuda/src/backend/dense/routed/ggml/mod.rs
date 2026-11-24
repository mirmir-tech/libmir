mod gathered;
mod grouped;
#[cfg(test)]
mod tests;

use mircuda::{DeviceBuffer, Stream, bf16};

use super::{
    super::RoutedExpertsConfig,
    weights::{GgmlExpertBank, GgmlExpertWeights},
};
use crate::{
    CudaBackend, Error, GatedActivation, Result,
    kernels::{
        GGML_GEMV_MAX_TOKENS, GgmlKernelType, GgmlMmqPlacement, GgmlOutputScaling, GgmlScaledRows,
    },
};

/// Shape of a layer's GGML expert banks.
#[derive(Clone, Copy, Debug)]
pub(in crate::backend) struct GgmlExpertGeometry {
    pub(in crate::backend) hidden: usize,
    /// Width of one expert.
    pub(in crate::backend) intermediate: usize,
    pub(in crate::backend) experts: RoutedExpertsConfig,
    pub(in crate::backend) activation: GatedActivation,
}

/// GGML expert banks for a fixed number of rows. Decode batches take one
/// int8 GEMV per (token, expert) selection; prefill chunks sort selections
/// by expert and run grouped int8 GEMMs over the banks. As with dense
/// projections, the bank's type picks the kernel, since a `Q4_K_M`
/// checkpoint mixes types across layers.
pub(in crate::backend) struct GgmlRoutedExperts {
    path: Path,
    stream: Stream,
    top_k: usize,
}

enum Path {
    Gathered(Box<gathered::Gathered>),
    Grouped(Box<grouped::Grouped>),
}

// The GGML expert kernels carry no `Debug`; the path and routing width name
// the plan.
impl std::fmt::Debug for GgmlRoutedExperts {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let path = match self.path {
            Path::Gathered(_) => "gathered",
            Path::Grouped(_) => "grouped",
        };
        formatter
            .debug_struct("GgmlRoutedExperts")
            .field("path", &path)
            .field("top_k", &self.top_k)
            .finish_non_exhaustive()
    }
}

/// The selections of one routed feed-forward call.
pub(super) struct Routing<'a> {
    pub(super) selected: &'a DeviceBuffer<u32>,
    pub(super) weights: &'a DeviceBuffer<bf16>,
    pub(super) top_k: usize,
}

impl GgmlRoutedExperts {
    pub(in crate::backend) fn new(
        backend: &CudaBackend,
        geometry: GgmlExpertGeometry,
        rows: usize,
    ) -> Result<Self> {
        let path = if rows <= GGML_GEMV_MAX_TOKENS {
            Path::Gathered(Box::new(gathered::Gathered::new(backend, geometry, rows)?))
        } else {
            Path::Grouped(Box::new(grouped::Grouped::new(backend, geometry, rows)?))
        };
        Ok(Self {
            path,
            stream: backend.stream().clone(),
            top_k: geometry.experts.top_k,
        })
    }

    pub(in crate::backend) fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        routing: &DeviceBuffer<bf16>,
        weights: &GgmlExpertWeights,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let routing = Routing {
            selected,
            weights: routing,
            top_k: self.top_k,
        };
        match &mut self.path {
            Path::Gathered(path) => path.execute(&self.stream, input, &routing, weights, output),
            Path::Grouped(path) => path.execute(&self.stream, input, &routing, weights, output),
        }
    }
}

/// Every GGML type whose blocks tile rows of `columns` values.
fn tiling(columns: usize) -> impl Iterator<Item = GgmlKernelType> {
    GgmlKernelType::all().filter(move |kind| {
        usize::try_from(kind.ggml_type().block_elements())
            .is_ok_and(|block| columns.is_multiple_of(block))
    })
}

/// Multiplies the products of `rows` selection rows, `columns` at
/// `placement` of each, by their experts' scales when `bank` has them.
#[allow(clippy::too_many_arguments)]
fn scale_rows(
    scaling: &GgmlOutputScaling,
    stream: &Stream,
    bank: GgmlExpertBank<'_>,
    data: &mut DeviceBuffer<bf16>,
    experts: &DeviceBuffer<u32>,
    (rows, columns): (usize, usize),
    placement: GgmlMmqPlacement,
) -> Result<()> {
    bank.scale().map_or(Ok(()), |scales| {
        let rows = GgmlScaledRows { experts, rows, columns, placement };
        scaling.expert_rows(stream, data, scales, rows)
    })
}

fn checked(left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right)
        .ok_or(Error::InvalidDecoderKernel("GGML routed expert size overflow"))
}
