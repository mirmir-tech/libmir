use mircuda::{DeviceBuffer, Stream, VariableGroupedBf16Plan, VariableGroupedBf16Spec, bf16};

use super::super::super::CudaBackend;
use crate::{
    Result,
    kernels::{SelectedDenseDispatch, SelectedDenseMoe, SelectedDenseReduceLaunch},
};

#[derive(Debug)]
pub(super) struct TensorCoreScratch {
    gate_up: VariableGroupedBf16Plan,
    down: VariableGroupedBf16Plan,
    compact_hidden: DeviceBuffer<bf16>,
    fused: DeviceBuffer<bf16>,
}

impl TensorCoreScratch {
    pub(super) fn new(backend: &CudaBackend, operation: &SelectedDenseMoe) -> Result<Self> {
        let spec = operation.spec();
        let routes = product(spec.tokens, spec.selected_count)?;
        let fused_width = product(spec.output_features, 2)?;
        let grouped =
            |n, k| VariableGroupedBf16Spec::new(spec.expert_count, spec.tokens, n, k, routes);
        Ok(Self {
            gate_up: VariableGroupedBf16Plan::new(
                &backend.inner.context,
                &backend.inner.stream,
                grouped(fused_width, spec.input_features)?,
            )?,
            down: VariableGroupedBf16Plan::new(
                &backend.inner.context,
                &backend.inner.stream,
                grouped(spec.input_features, spec.output_features)?,
            )?,
            compact_hidden: backend
                .inner
                .pool
                .allocate(&backend.inner.stream, product(routes, spec.input_features)?)?,
            fused: backend
                .inner
                .pool
                .allocate(&backend.inner.stream, product(routes, fused_width)?)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn execute(
        &mut self,
        operation: &SelectedDenseMoe,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        selected: &DeviceBuffer<u32>,
        routing: &DeviceBuffer<bf16>,
        gate_up_weight: &DeviceBuffer<bf16>,
        gate_up_bias: &DeviceBuffer<bf16>,
        down_weight: &DeviceBuffer<bf16>,
        down_bias: &DeviceBuffer<bf16>,
        intermediate: &mut DeviceBuffer<bf16>,
        partial: &mut DeviceBuffer<f32>,
        output: &mut DeviceBuffer<bf16>,
        dispatch: &SelectedDenseDispatch<'_>,
    ) -> Result<()> {
        let spec = operation.spec();
        operation.compact_expert_input(stream, input, dispatch, &mut self.compact_hidden)?;
        operation.fill_expert_bias(
            stream,
            dispatch,
            gate_up_bias,
            &mut self.fused,
            product(spec.output_features, 2)?,
            spec.gate_bias || spec.up_bias,
        )?;
        self.gate_up.execute(
            stream,
            &self.compact_hidden,
            gate_up_weight,
            &*dispatch.counts,
            &*dispatch.offsets,
            &mut self.fused,
            1.0,
        )?;
        operation.activate_compact(stream, &self.fused, intermediate)?;
        operation.fill_expert_bias(
            stream,
            dispatch,
            down_bias,
            &mut self.compact_hidden,
            spec.input_features,
            spec.down_bias,
        )?;
        self.down.execute(
            stream,
            intermediate,
            down_weight,
            &*dispatch.counts,
            &*dispatch.offsets,
            &mut self.compact_hidden,
            1.0,
        )?;
        operation.route_compact(
            stream,
            &self.compact_hidden,
            dispatch,
            &mut SelectedDenseReduceLaunch {
                input: intermediate,
                selected,
                routing,
                weight: down_weight,
                bias: down_bias,
                partial,
                output,
            },
        )
    }
}

fn product(left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right).ok_or(crate::Error::InvalidDecoderKernel(
        "dense grouped tensor-core scratch size overflow",
    ))
}
