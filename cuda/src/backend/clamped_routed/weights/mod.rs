use models::weights::{
    DecoderBoundaryBindings, RoutedDecoderLayerBindings, TensorBinding, TensorStorage,
};

use super::{
    ClampedRoutedConfig,
    layout::ClampedRoutedLayout,
    projection::{
        ClampedRoutedBoundaryProjection, ClampedRoutedLinearWeight, ClampedRoutedQkvWeight,
    },
    validation::validate_common,
};
use crate::{AffineQuantizedWeight, CudaBackend, CudaTensor, CudaTensorSet, Error, Result};

mod mlx;
mod native;

#[derive(Clone)]
pub(super) struct ClampedRoutedLayerWeights {
    pub input_norm: CudaTensor,
    pub qkv: ClampedRoutedQkvWeight,
    pub output: ClampedRoutedLinearWeight,
    pub output_bias: CudaTensor,
    pub sinks: CudaTensor,
    pub post_norm: CudaTensor,
    pub router: ClampedRoutedLinearWeight,
    pub router_bias: CudaTensor,
    pub experts: ClampedRoutedExpertWeights,
}

#[derive(Clone)]
pub(super) enum ClampedRoutedExpertWeights {
    Native(Box<NativeExpertWeights>),
    Mlx(Box<MlxExpertWeights>),
}

#[derive(Clone)]
pub(super) struct NativeExpertWeights {
    pub gate_up_blocks: CudaTensor,
    pub gate_up_scales: CudaTensor,
    pub gate_up_bias: CudaTensor,
    pub down_blocks: CudaTensor,
    pub down_scales: CudaTensor,
    pub down_bias: CudaTensor,
}

#[derive(Clone)]
pub(super) struct MlxExpertWeights {
    pub gate_blocks: CudaTensor,
    pub gate_scales: CudaTensor,
    pub gate_bias: CudaTensor,
    pub up_blocks: CudaTensor,
    pub up_scales: CudaTensor,
    pub up_bias: CudaTensor,
    pub down_blocks: CudaTensor,
    pub down_scales: CudaTensor,
    pub down_bias: CudaTensor,
}

pub(super) fn boundary(
    layout: ClampedRoutedLayout,
    tensors: &CudaTensorSet,
    bindings: DecoderBoundaryBindings<'_>,
    config: ClampedRoutedConfig,
) -> Result<(ClampedRoutedBoundaryProjection, CudaTensor, ClampedRoutedBoundaryProjection)> {
    let norm = tensor(tensors, &bindings.final_norm.source)?;
    match layout {
        ClampedRoutedLayout::Native => Ok((
            ClampedRoutedBoundaryProjection::Native(tensor(tensors, &bindings.embedding.source)?),
            norm,
            ClampedRoutedBoundaryProjection::Native(tensor(tensors, &bindings.output.source)?),
        )),
        ClampedRoutedLayout::Mlx => {
            let embedding = affine(tensors, bindings.embedding)?;
            let output = affine(tensors, bindings.output)?;
            embedding.infer_config(1, config.hidden, config.vocab)?;
            output.infer_config(1, config.hidden, config.vocab)?;
            Ok((
                ClampedRoutedBoundaryProjection::Mlx(embedding),
                norm,
                ClampedRoutedBoundaryProjection::Mlx(output),
            ))
        },
    }
}

fn affine(tensors: &CudaTensorSet, binding: &TensorBinding) -> Result<AffineQuantizedWeight> {
    AffineQuantizedWeight::load_binding(tensors, binding)
}

pub(super) fn layer(
    backend: &CudaBackend,
    layout: ClampedRoutedLayout,
    config: ClampedRoutedConfig,
    tensors: &CudaTensorSet,
    bindings: RoutedDecoderLayerBindings<'_>,
) -> Result<ClampedRoutedLayerWeights> {
    let input_norm = tensor(tensors, &bindings.input_norm.source)?;
    let q_bias = binding_bias(tensors, bindings.query)?;
    let k_bias = binding_bias(tensors, bindings.key)?;
    let v_bias = binding_bias(tensors, bindings.value)?;
    let output_bias = binding_bias(tensors, bindings.attention_output)?;
    let sinks = tensor(tensors, &bindings.attention_sinks.source)?;
    let post_norm = tensor(tensors, &bindings.post_attention_norm.source)?;
    let router_bias = binding_bias(tensors, bindings.router)?;
    validate_common(
        config,
        [&input_norm, &q_bias, &k_bias, &v_bias, &output_bias, &sinks, &post_norm, &router_bias],
    )?;
    let (qkv, output, router, experts) = match layout {
        ClampedRoutedLayout::Native => native::load(backend, config, tensors, bindings)?,
        ClampedRoutedLayout::Mlx => mlx::load(config, tensors, bindings)?,
    };
    Ok(ClampedRoutedLayerWeights {
        input_norm,
        qkv: ClampedRoutedQkvWeight {
            projections: qkv,
            biases: [q_bias, k_bias, v_bias],
        },
        output,
        output_bias,
        sinks,
        post_norm,
        router,
        router_bias,
        experts,
    })
}

fn binding_bias(tensors: &CudaTensorSet, binding: &TensorBinding) -> Result<CudaTensor> {
    let name = match &binding.storage {
        TensorStorage::Dense { bias, .. } | TensorStorage::BlockQuantized { bias, .. } => {
            bias.as_deref()
        },
        TensorStorage::AffineQuantized { output_bias, .. } => output_bias.as_deref(),
        TensorStorage::Auxiliary { .. } => None,
    }
    .ok_or_else(|| Error::MissingTensor(format!("bias for logical tensor {}", binding.source)))?;
    tensor(tensors, name)
}

pub(super) fn tensor(tensors: &CudaTensorSet, name: &str) -> Result<CudaTensor> {
    tensors.get(name).cloned().ok_or_else(|| Error::MissingTensor(name.into()))
}
