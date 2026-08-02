use std::time::{Duration, Instant};

use mircuda::{DeviceBuffer, ScaledFp8Scale, bf16};

use super::{
    CudaBackend, DirectFp8Activation, DirectFp8CheckpointWeight, DirectFp8Format, DirectFp8Scale,
    DirectFp8Spec, Error, Result,
    candidate::{Candidate, tensor_core_admitted},
};
use crate::{
    PlanSource,
    backend::tuning::{
        DirectFp8ProjectionExecution, DirectFp8ScaleDType,
        DirectFp8WeightScale as ProfileWeightScale, QuantizedProfileExecution,
        QuantizedProfileRequest,
    },
};

mod measure;

pub(super) fn prepare(
    backend: &CudaBackend,
    weight: &DirectFp8CheckpointWeight,
    spec: DirectFp8Spec,
    tensor_core_scale: Option<ScaledFp8Scale>,
    identity_scale: Option<&DeviceBuffer<f32>>,
) -> Result<Candidate> {
    if !tensor_core_admitted(backend, spec, tensor_core_scale) {
        return Candidate::new(
            backend,
            spec,
            tensor_core_scale,
            weight.bias.is_some(),
            DirectFp8ProjectionExecution::Portable,
        );
    }
    let scale_dtype = tensor_core_scale.map(|scale| match scale {
        ScaledFp8Scale::F32 => DirectFp8ScaleDType::F32,
        ScaledFp8Scale::Bf16 => DirectFp8ScaleDType::Bf16,
    });
    let request = profile_request(spec, scale_dtype, weight.bias.is_some())?;
    if let Some((QuantizedProfileExecution::DirectFp8(execution), source)) =
        backend.auto_tuner().lookup_quantized(request)
    {
        match Candidate::new(backend, spec, tensor_core_scale, weight.bias.is_some(), execution) {
            Ok(candidate) => {
                trace_selection(request, execution, source, None);
                return Ok(candidate);
            },
            Err(error) => tracing::warn!(
                ?execution,
                %error,
                "cached direct FP8 candidate is unavailable; using format fallback"
            ),
        }
    }
    let fallback_execution = if spec.format == DirectFp8Format::E5M2 {
        DirectFp8ProjectionExecution::Portable
    } else {
        DirectFp8ProjectionExecution::TensorCore
    };
    let fallback = Candidate::new(
        backend,
        spec,
        tensor_core_scale,
        weight.bias.is_some(),
        fallback_execution,
    )?;
    if !backend.auto_tuner().claim_quantized(request) {
        return Ok(fallback);
    }
    match tune(backend, request, spec, tensor_core_scale, weight, identity_scale, fallback) {
        Ok(candidate) => Ok(candidate),
        Err(error) => {
            backend.auto_tuner().abandon_quantized(request);
            tracing::warn!(?request, %error, ?fallback_execution, "CUDA direct FP8 tuning failed; retaining fallback");
            Candidate::new(
                backend,
                spec,
                tensor_core_scale,
                weight.bias.is_some(),
                fallback_execution,
            )
        },
    }
}

fn profile_request(
    spec: DirectFp8Spec,
    scale_dtype: Option<DirectFp8ScaleDType>,
    bias: bool,
) -> Result<QuantizedProfileRequest> {
    match (spec.activation, spec.scale) {
        (DirectFp8Activation::DynamicE4M3Token, DirectFp8Scale::OutputChannel) => {
            Ok(QuantizedProfileRequest::direct_fp8_dynamic_e4m3(
                spec.tokens,
                spec.input_features,
                spec.output_features,
                scale_dtype
                    .ok_or(Error::InvalidExecutionPlan("dynamic E4M3 tuning scale is missing"))?,
                bias,
            ))
        },
        (DirectFp8Activation::StaticE4M3Tensor, scale) => {
            let weight_scale = match scale {
                DirectFp8Scale::Tensor => ProfileWeightScale::Tensor,
                DirectFp8Scale::OutputChannel => ProfileWeightScale::OutputChannel,
                DirectFp8Scale::BlockGrid { .. } => {
                    return Err(Error::InvalidExecutionPlan(
                        "direct FP8 Tensor Core profile does not accept a block scale grid",
                    ));
                },
            };
            Ok(QuantizedProfileRequest::direct_fp8_static_e4m3(
                spec.tokens,
                spec.input_features,
                spec.output_features,
                weight_scale,
                scale_dtype
                    .ok_or(Error::InvalidExecutionPlan("static E4M3 tuning scale is missing"))?,
                bias,
            ))
        },
        (DirectFp8Activation::Bf16, DirectFp8Scale::Tensor)
            if spec.format == DirectFp8Format::E5M2 && scale_dtype.is_none() =>
        {
            Ok(QuantizedProfileRequest::direct_fp8_bf16_e5m2_weight_only(
                spec.tokens,
                spec.input_features,
                spec.output_features,
                bias,
            ))
        },
        _ => Err(Error::InvalidExecutionPlan(
            "direct FP8 Tensor Core profile contract is unavailable",
        )),
    }
}

#[allow(clippy::too_many_arguments)]
fn tune(
    backend: &CudaBackend,
    request: QuantizedProfileRequest,
    spec: DirectFp8Spec,
    tensor_core_scale: Option<ScaledFp8Scale>,
    weight: &DirectFp8CheckpointWeight,
    identity_scale: Option<&DeviceBuffer<f32>>,
    fallback: Candidate,
) -> Result<Candidate> {
    let started = Instant::now();
    let alternative = match fallback.execution {
        DirectFp8ProjectionExecution::Portable => DirectFp8ProjectionExecution::TensorCore,
        DirectFp8ProjectionExecution::TensorCore => DirectFp8ProjectionExecution::Portable,
    };
    let mut candidates = vec![fallback];
    match Candidate::new(backend, spec, tensor_core_scale, weight.bias.is_some(), alternative) {
        Ok(candidate) => candidates.push(candidate),
        Err(error) => tracing::debug!(%error, "alternate direct FP8 candidate is unavailable"),
    }
    let input = sample_input(backend, spec.input_elements()?)?;
    let mut output = backend
        .pool()
        .allocate_zeroed::<bf16>(backend.stream(), spec.output_elements()?)?;
    measure::retain_compatible(
        backend, request, &mut candidates, &input, weight, identity_scale, &mut output,
    )?;
    let (selected, average, measured) = measure::select(
        backend, spec, &mut candidates, &input, weight, identity_scale, &mut output,
    )?;
    let selected = candidates.swap_remove(selected);
    backend.auto_tuner().record_quantized(
        request,
        QuantizedProfileExecution::DirectFp8(selected.execution),
        average,
        started.elapsed().max(measured),
    );
    trace_selection(request, selected.execution, PlanSource::MeasuredStartup, Some(average));
    Ok(selected)
}

fn sample_input(backend: &CudaBackend, elements: usize) -> Result<DeviceBuffer<bf16>> {
    const PATTERN: [f32; 16] = [
        -1.0, -0.75, -0.5, -0.25, -0.125, -0.0625, -0.03125, 0.0, 0.03125, 0.0625, 0.125, 0.25,
        0.5, 0.75, 1.0, 0.375,
    ];
    let values = (0..elements)
        .map(|index| bf16::from_f32(PATTERN[index % PATTERN.len()]))
        .collect::<Vec<_>>();
    let mut host = backend.context().allocate_pinned(elements)?;
    host.copy_from_slice(&values)?;
    let mut input = backend.pool().allocate(backend.stream(), elements)?;
    backend.stream().copy_to_device(&mut host, &mut input)?;
    Ok(input)
}

fn trace_selection(
    request: QuantizedProfileRequest,
    execution: DirectFp8ProjectionExecution,
    source: PlanSource,
    average: Option<Duration>,
) {
    tracing::info!(
        target: "libmir::cuda::tuning",
        ?request,
        ?execution,
        ?source,
        average_us = average.map(|value| value.as_secs_f64() * 1_000_000.0),
        "selected CUDA direct FP8 projection execution"
    );
}
