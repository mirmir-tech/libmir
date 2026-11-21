mod gathered;
mod grouped;

use mircuda::{DeviceBuffer, Stream, bf16};

use super::{super::RoutedExpertsConfig, weights::GgmlExpertWeights};
use crate::{
    CudaBackend, Error, GatedActivation, Result,
    kernels::{GGML_GEMV_MAX_TOKENS, GgmlKernelType},
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
    GgmlKernelType::ALL.into_iter().filter(move |kind| {
        usize::try_from(kind.ggml_type().block_elements())
            .is_ok_and(|block| columns.is_multiple_of(block))
    })
}

fn checked(left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right)
        .ok_or(Error::InvalidDecoderKernel("GGML routed expert size overflow"))
}
