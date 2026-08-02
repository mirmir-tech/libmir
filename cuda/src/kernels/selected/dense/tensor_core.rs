use mircuda::{DeviceBuffer, LaunchConfig, Module, Stream, TypedKernel, bf16, cuda_export};

use super::{SelectedDenseDispatch, SelectedDenseMoe, SelectedDenseReduceLaunch};
use crate::{
    Result,
    kernels::geometry::{narrow, product, require},
};

cuda_export!(CompactKernel = "libmir_cuda_selected_dense_compact_bf16"(
    input: &DeviceBuffer<bf16>, assignments: &DeviceBuffer<u32>,
    output: &mut DeviceBuffer<bf16>, features: u32,
    selected_count: u32, routes: u32,
));
cuda_export!(BiasKernel = "libmir_cuda_selected_dense_fill_bias_bf16"(
    experts: &DeviceBuffer<u32>, bias: &DeviceBuffer<bf16>,
    output: &mut DeviceBuffer<bf16>, features: u32, routes: u32,
    has_bias: u32,
));
cuda_export!(ActivateKernel = "libmir_cuda_selected_dense_activate_compact_bf16"(
    fused: &DeviceBuffer<bf16>, output: &mut DeviceBuffer<bf16>,
    features: u32, routes: u32, activation: u32, alpha: f32,
    limit: f32, up_shift: f32,
));
cuda_export!(RouteKernel = "libmir_cuda_selected_dense_route_compact_bf16"(
    input: &DeviceBuffer<bf16>, assignments: &DeviceBuffer<u32>,
    routing: &DeviceBuffer<bf16>, partial: &mut DeviceBuffer<f32>,
    features: u32, routes: u32,
));

#[derive(Clone, Debug)]
pub(super) struct TensorCoreKernels {
    compact: TypedKernel<CompactKernel>,
    bias: TypedKernel<BiasKernel>,
    activate: TypedKernel<ActivateKernel>,
    route: TypedKernel<RouteKernel>,
}

impl TensorCoreKernels {
    pub(super) fn new(module: &Module) -> Result<Self> {
        Ok(Self {
            compact: module.kernel()?,
            bias: module.kernel()?,
            activate: module.kernel()?,
            route: module.kernel()?,
        })
    }
}

impl SelectedDenseMoe {
    pub fn compact_expert_input(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        dispatch: &SelectedDenseDispatch<'_>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let spec = self.spec();
        let routes = product(spec.tokens, spec.selected_count)?;
        require("dense compact input", product(spec.tokens, spec.input_features)?, input.len())?;
        require("dense compact output", product(routes, spec.input_features)?, output.len())?;
        Ok(self.tensor_core.compact.launch(
            stream,
            linear(product(routes, spec.input_features)?)?,
            (
                input,
                &*dispatch.assignments,
                output,
                narrow(spec.input_features)?,
                narrow(spec.selected_count)?,
                narrow(routes)?,
            ),
        )?)
    }

    pub fn fill_expert_bias(
        &self,
        stream: &Stream,
        dispatch: &SelectedDenseDispatch<'_>,
        bias: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        features: usize,
        present: bool,
    ) -> Result<()> {
        let routes = product(self.spec().tokens, self.spec().selected_count)?;
        require("dense compact bias output", product(routes, features)?, output.len())?;
        if present {
            require(
                "dense compact expert bias",
                product(self.spec().expert_count, features)?,
                bias.len(),
            )?;
        }
        Ok(self.tensor_core.bias.launch(
            stream,
            linear(product(routes, features)?)?,
            (
                &*dispatch.experts,
                bias,
                output,
                narrow(features)?,
                narrow(routes)?,
                u32::from(present),
            ),
        )?)
    }

    pub fn activate_compact(
        &self,
        stream: &Stream,
        fused: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let spec = self.spec();
        let routes = product(spec.tokens, spec.selected_count)?;
        require(
            "dense compact fused",
            product(product(routes, spec.output_features)?, 2)?,
            fused.len(),
        )?;
        require("dense compact activation", product(routes, spec.output_features)?, output.len())?;
        Ok(self.tensor_core.activate.launch(
            stream,
            linear(product(routes, spec.output_features)?)?,
            (
                fused,
                output,
                narrow(spec.output_features)?,
                narrow(routes)?,
                spec.activation.kind,
                spec.activation.alpha,
                spec.activation.limit,
                spec.activation.up_shift,
            ),
        )?)
    }

    pub fn route_compact(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        dispatch: &SelectedDenseDispatch<'_>,
        launch: &mut SelectedDenseReduceLaunch<'_>,
    ) -> Result<()> {
        let spec = self.spec();
        let routes = product(spec.tokens, spec.selected_count)?;
        require("dense compact projected", product(routes, spec.input_features)?, input.len())?;
        self.tensor_core.route.launch(
            stream,
            linear(product(routes, spec.input_features)?)?,
            (
                input,
                &*dispatch.assignments,
                launch.routing,
                &mut *launch.partial,
                narrow(spec.input_features)?,
                narrow(routes)?,
            ),
        )?;
        self.finalize(stream, launch)
    }
}

fn linear(elements: usize) -> Result<LaunchConfig> {
    Ok(LaunchConfig {
        grid: (narrow(elements.div_ceil(256))?, 1, 1),
        block: (256, 1, 1),
        shared_memory_bytes: 0,
    })
}
