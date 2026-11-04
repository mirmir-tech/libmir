//! Marlin W4A16 routed experts with an ungated `relu²` activation:
//! `down(relu(up(x))²)`, weighted by the routing weights.

use std::sync::{Arc, Mutex};

use mircuda::{Context, DeviceBuffer, MarlinNvFp4MoeSpec, MarlinNvFp4ThreadConfig, Stream, bf16};

use super::{
    moe::{ProjectionOutput, execute_projection, prepare_routes, product},
    scratch::{MarlinNvFp4Scratch, MarlinNvFp4ScratchConfig, MarlinRouteBlock},
};
use crate::{
    CudaBackend, Error, Result,
    backend::linear::nvfp4::bank::MarlinNvFp4Bank,
    kernels::{ElementwiseBf16, ReluSquaredBf16},
};

#[derive(Debug)]
pub(in crate::backend) struct MarlinNvFp4UngatedMoeBf16 {
    up: MarlinNvFp4Bank,
    down: MarlinNvFp4Bank,
    scratch: Arc<Mutex<MarlinNvFp4Scratch>>,
    activation: ReluSquaredBf16,
    reduce: ElementwiseBf16,
    context: Context,
    stream: Stream,
    tokens: usize,
    top_k: usize,
    thread_config: MarlinNvFp4ThreadConfig,
}

impl CudaBackend {
    pub(in crate::backend) fn prepare_marlin_nvfp4_ungated_moe_bf16(
        &self,
        tokens: usize,
        top_k: usize,
        thread_config: MarlinNvFp4ThreadConfig,
        [up, down]: [MarlinNvFp4Bank; 2],
    ) -> Result<MarlinNvFp4UngatedMoeBf16> {
        let experts = up.config.experts;
        let width = up.config.input_features;
        let intermediate = up.config.output_features;
        if down.config.input_features != intermediate
            || down.config.output_features != width
            || down.config.experts != experts
            || tokens == 0
            || top_k == 0
        {
            return Err(Error::InvalidNvFp4("incompatible ungated Marlin MoE banks"));
        }
        Ok(MarlinNvFp4UngatedMoeBf16 {
            up,
            down,
            scratch: self.marlin_nvfp4_scratch(MarlinNvFp4ScratchConfig {
                tokens,
                top_k,
                experts,
                route_block: MarlinRouteBlock::from(thread_config),
                hidden: width,
                intermediate,
                padded_hidden: width,
                padded_intermediate: intermediate,
            })?,
            activation: ReluSquaredBf16::compile(&self.inner.compiler)?,
            reduce: ElementwiseBf16::compile(&self.inner.compiler, width)?,
            context: self.inner.context.clone(),
            stream: self.inner.stream.clone(),
            tokens,
            top_k,
            thread_config,
        })
    }
}

impl MarlinNvFp4UngatedMoeBf16 {
    /// `input` holds `tokens` rows at the expert width; `output` receives the
    /// routing-weighted sum of the selected experts.
    pub(in crate::backend) fn execute(
        &self,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        routing: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let mut scratch = self
            .scratch
            .lock()
            .map_err(|_| Error::InvalidExecutionPlan("Marlin scratch lock is poisoned"))?;
        let experts = self.up.config.experts;
        let width = self.up.config.input_features;
        let intermediate = self.up.config.output_features;
        let first = MarlinNvFp4MoeSpec::new(
            experts,
            self.tokens,
            self.top_k,
            intermediate,
            width,
            self.thread_config,
        )?;
        prepare_routes(&self.context, &self.stream, first, selected, routing, &mut scratch)?;
        execute_projection(
            &self.context,
            &self.stream,
            first,
            input,
            &self.up,
            &mut scratch,
            ProjectionOutput::Intermediate,
        )?;
        // The activation runs in place on the exactly sized intermediate.
        let projected = scratch.intermediate.clone();
        self.activation.execute(
            &self.stream,
            &projected,
            &mut scratch.intermediate,
            product(first.assignments(), intermediate)?,
        )?;
        let second = MarlinNvFp4MoeSpec::new(
            experts,
            first.assignments(),
            1,
            width,
            intermediate,
            self.thread_config,
        )?;
        let activated = scratch.intermediate.clone();
        execute_projection(
            &self.context,
            &self.stream,
            second,
            &activated,
            &self.down,
            &mut scratch,
            ProjectionOutput::Down,
        )?;
        self.reduce.weighted_reduce_batch(
            &self.stream, &scratch.down, routing, output, self.top_k, self.tokens,
        )
    }
}
