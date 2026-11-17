//! The weighted sum of each token's expert-order rows after a grouped down
//! projection, through the inverse of the dispatch's assignments.

use mircuda::{Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export};

use super::{
    super::super::geometry::{narrow, product, require},
    module,
};
use crate::{Error, Result};

cuda_export!(InvertKernel = "libmir_cuda_ggml_mmq_invert"(
    assignments: &DeviceBuffer<u32>, rows_of_routes: &mut DeviceBuffer<u32>, routes: u32,
));
cuda_export!(CombineKernel = "libmir_cuda_ggml_mmq_combine_bf16"(
    rows: &DeviceBuffer<bf16>, rows_of_routes: &DeviceBuffer<u32>, selected: &DeviceBuffer<u32>,
    routing: &DeviceBuffer<bf16>, output: &mut DeviceBuffer<bf16>, tokens: u32, top_k: u32,
    features: u32, experts: u32,
));

/// The routed sums of `tokens` tokens of `top_k` selections each.
#[derive(Debug)]
pub struct GgmlRouteCombine {
    invert: TypedKernel<InvertKernel>,
    combine: TypedKernel<CombineKernel>,
    rows_of_routes: DeviceBuffer<u32>,
    tokens: usize,
    top_k: usize,
    features: usize,
    experts: usize,
}

impl GgmlRouteCombine {
    pub fn compile(
        compiler: &Compiler,
        rows_of_routes: DeviceBuffer<u32>,
        tokens: usize,
        top_k: usize,
        features: usize,
        experts: usize,
    ) -> Result<Self> {
        if !features.is_multiple_of(8) || rows_of_routes.len() < product(tokens, top_k)? {
            return Err(Error::InvalidDecoderKernel("invalid GGML route combine geometry"));
        }
        let module = module(compiler)?;
        Ok(Self {
            invert: module.kernel()?,
            combine: module.kernel()?,
            rows_of_routes,
            tokens,
            top_k,
            features,
            experts,
        })
    }

    /// `output[t] = sum_k routing[t, k] * rows[row(t, k)]`, where
    /// `assignments[row]` is the selection `t * top_k + k` of each row.
    pub fn execute(
        &mut self,
        stream: &Stream,
        rows: &DeviceBuffer<bf16>,
        assignments: &DeviceBuffer<u32>,
        selected: &DeviceBuffer<u32>,
        routing: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let routes = product(self.tokens, self.top_k)?;
        require("GGML combine rows", product(routes, self.features)?, rows.len())?;
        require("GGML combine assignments", routes, assignments.len())?;
        require("GGML combine selections", routes, selected.len())?;
        require("GGML combine routing", routes, routing.len())?;
        require("GGML combine output", product(self.tokens, self.features)?, output.len())?;
        let linear = |elements: usize| -> Result<LaunchConfig> {
            Ok(LaunchConfig {
                grid: (narrow(elements.div_ceil(256))?, 1, 1),
                block: (256, 1, 1),
                shared_memory_bytes: 0,
            })
        };
        self.invert.launch(
            stream,
            linear(routes)?,
            (assignments, &mut self.rows_of_routes, narrow(routes)?),
        )?;
        self.combine.launch(
            stream,
            linear(product(self.tokens, self.features / 8)?)?,
            (
                rows,
                &self.rows_of_routes,
                selected,
                routing,
                output,
                narrow(self.tokens)?,
                narrow(self.top_k)?,
                narrow(self.features)?,
                narrow(self.experts)?,
            ),
        )?;
        Ok(())
    }
}
