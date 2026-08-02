use mircuda::{DeviceBuffer, LaunchConfig, Module, Stream, TypedKernel, bf16, cuda_export};

use super::{SelectedDenseGateLaunch, SelectedDenseMoe, SelectedDenseReduceLaunch};
use crate::{Result, kernels::geometry::narrow};

cuda_export!(ClearKernel = "libmir_cuda_selected_dense_dispatch_clear"(
    counts: &mut DeviceBuffer<u32>, cursors: &mut DeviceBuffer<u32>,
    experts: u32,
));
cuda_export!(CountKernel = "libmir_cuda_selected_dense_dispatch_count"(
    selected: &DeviceBuffer<u32>, counts: &mut DeviceBuffer<u32>,
    assignments: u32, experts: u32,
));
cuda_export!(PrefixKernel = "libmir_cuda_selected_dense_dispatch_prefix"(
    counts: &DeviceBuffer<u32>, offsets: &mut DeviceBuffer<u32>,
    cursors: &mut DeviceBuffer<u32>, experts: u32,
));
cuda_export!(ScatterKernel = "libmir_cuda_selected_dense_dispatch_scatter"(
    selected: &DeviceBuffer<u32>, offsets: &DeviceBuffer<u32>,
    cursors: &mut DeviceBuffer<u32>, assignments: &mut DeviceBuffer<u32>,
    experts_out: &mut DeviceBuffer<u32>, count: u32, experts: u32,
));
cuda_export!(GatedKernel = "libmir_cuda_selected_dense_gated_expert_major_bf16"(
    input: &DeviceBuffer<bf16>, assignments: &DeviceBuffer<u32>,
    experts: &DeviceBuffer<u32>, gate_up_weight: &DeviceBuffer<bf16>,
    gate_up_bias: &DeviceBuffer<bf16>, output: &mut DeviceBuffer<bf16>,
    input_features: u32, output_features: u32, selected_count: u32,
    has_gate_bias: u32, has_up_bias: u32, activation: u32, alpha: f32,
    limit: f32, up_shift: f32,
));
cuda_export!(ProjectKernel = "libmir_cuda_selected_dense_project_expert_major_bf16"(
    input: &DeviceBuffer<bf16>, assignments: &DeviceBuffer<u32>,
    experts: &DeviceBuffer<u32>, routing: &DeviceBuffer<bf16>,
    weight: &DeviceBuffer<bf16>, bias: &DeviceBuffer<bf16>,
    partial: &mut DeviceBuffer<f32>, input_features: u32,
    output_features: u32, selected_count: u32, has_bias: u32,
));

pub struct SelectedDenseDispatch<'a> {
    pub counts: &'a mut DeviceBuffer<u32>,
    pub offsets: &'a mut DeviceBuffer<u32>,
    pub cursors: &'a mut DeviceBuffer<u32>,
    pub assignments: &'a mut DeviceBuffer<u32>,
    pub experts: &'a mut DeviceBuffer<u32>,
}

#[derive(Clone, Debug)]
pub(super) struct ExpertMajorKernels {
    clear: TypedKernel<ClearKernel>,
    count: TypedKernel<CountKernel>,
    prefix: TypedKernel<PrefixKernel>,
    scatter: TypedKernel<ScatterKernel>,
    gated: TypedKernel<GatedKernel>,
    project: TypedKernel<ProjectKernel>,
}

impl ExpertMajorKernels {
    pub(super) fn new(module: &Module) -> Result<Self> {
        Ok(Self {
            clear: module.kernel()?,
            count: module.kernel()?,
            prefix: module.kernel()?,
            scatter: module.kernel()?,
            gated: module.kernel()?,
            project: module.kernel()?,
        })
    }
}

impl SelectedDenseMoe {
    pub fn prepare_expert_major(
        &self,
        stream: &Stream,
        selected: &DeviceBuffer<u32>,
        dispatch: &mut SelectedDenseDispatch<'_>,
    ) -> Result<()> {
        self.validate_dispatch(selected, dispatch)?;
        let spec = self.spec;
        let assignments = spec.tokens * spec.selected_count;
        self.expert_major.clear.launch(
            stream,
            linear(spec.expert_count)?,
            (&mut *dispatch.counts, &mut *dispatch.cursors, narrow(spec.expert_count)?),
        )?;
        self.expert_major.count.launch(
            stream,
            linear(assignments)?,
            (
                selected,
                &mut *dispatch.counts,
                narrow(assignments)?,
                narrow(spec.expert_count)?,
            ),
        )?;
        self.expert_major.prefix.launch(
            stream,
            LaunchConfig {
                grid: (1, 1, 1),
                block: (1, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                &*dispatch.counts,
                &mut *dispatch.offsets,
                &mut *dispatch.cursors,
                narrow(spec.expert_count)?,
            ),
        )?;
        Ok(self.expert_major.scatter.launch(
            stream,
            linear(assignments)?,
            (
                selected,
                &*dispatch.offsets,
                &mut *dispatch.cursors,
                &mut *dispatch.assignments,
                &mut *dispatch.experts,
                narrow(assignments)?,
                narrow(spec.expert_count)?,
            ),
        )?)
    }

    pub fn gated_expert_major(
        &self,
        stream: &Stream,
        launch: &mut SelectedDenseGateLaunch<'_>,
        dispatch: &SelectedDenseDispatch<'_>,
    ) -> Result<()> {
        self.validate_gated(launch)?;
        let spec = self.spec;
        Ok(self.expert_major.gated.launch(
            stream,
            expert_grid(spec.output_features, spec.tokens * spec.selected_count, 16)?,
            (
                launch.input,
                &*dispatch.assignments,
                &*dispatch.experts,
                launch.gate_weight,
                launch.gate_bias,
                &mut *launch.output,
                narrow(spec.input_features)?,
                narrow(spec.output_features)?,
                narrow(spec.selected_count)?,
                u32::from(spec.gate_bias),
                u32::from(spec.up_bias),
                spec.activation.kind,
                spec.activation.alpha,
                spec.activation.limit,
                spec.activation.up_shift,
            ),
        )?)
    }

    pub fn reduce_expert_major(
        &self,
        stream: &Stream,
        launch: &mut SelectedDenseReduceLaunch<'_>,
        dispatch: &SelectedDenseDispatch<'_>,
    ) -> Result<()> {
        self.validate_reduce(launch)?;
        let spec = self.spec;
        self.expert_major.project.launch(
            stream,
            expert_grid(spec.input_features, spec.tokens * spec.selected_count, 8)?,
            (
                launch.input,
                &*dispatch.assignments,
                &*dispatch.experts,
                launch.routing,
                launch.weight,
                launch.bias,
                &mut *launch.partial,
                narrow(spec.output_features)?,
                narrow(spec.input_features)?,
                narrow(spec.selected_count)?,
                u32::from(spec.down_bias),
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

fn expert_grid(rows: usize, assignments: usize, shared_rows: usize) -> Result<LaunchConfig> {
    Ok(LaunchConfig {
        grid: (narrow(rows.div_ceil(32))?, 1, narrow(assignments)?),
        block: (32, 8, 1),
        shared_memory_bytes: narrow(shared_rows * 32 * size_of::<f32>())?,
    })
}
