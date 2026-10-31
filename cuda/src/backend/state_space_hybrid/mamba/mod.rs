//! Mamba-2 layer: `RMSNorm` → FP8 `in_proj` → convolution, selective scan and
//! gated norm → FP8 `out_proj` → residual, then the optional feed-forward.

mod execution;
mod scratch;

pub use execution::CudaMamba2HybridExecution;
use models::{
    semantic::{DecoderLayerSpec, MixerSpec, StateSpaceSpec, StateSpaceStateDTypeSpec},
    weights::{StateSpaceMixerBindings, TensorBinding, TensorStorage},
};
pub(in crate::backend) use scratch::{MixerScratch, MixerScratchKey};

use super::NormedFeedForward;
use crate::{
    CudaBackend, CudaMamba2State, CudaTensor, CudaTensorSet, DirectFp8CheckpointWeight, Error,
    ExecutionPhase, Result,
    backend::feed_forward::LayerNormConfig,
    kernels::{Mamba2Geometry, Mamba2Parameters},
};

#[derive(Clone, Debug)]
pub struct CudaMamba2HybridLayer {
    backend: CudaBackend,
    norm: LayerNormConfig,
    geometry: Mamba2Geometry,
    input_norm: CudaTensor,
    input_projection: DirectFp8CheckpointWeight,
    output_projection: DirectFp8CheckpointWeight,
    mixer: MixerWeights,
    feed_forward: Option<NormedFeedForward>,
}

#[derive(Clone, Debug)]
struct MixerWeights {
    convolution_weight: CudaTensor,
    convolution_bias: Option<CudaTensor>,
    a_log: CudaTensor,
    skip: CudaTensor,
    dt_bias: CudaTensor,
    norm_weight: CudaTensor,
}

impl CudaMamba2HybridLayer {
    pub(in crate::backend) fn load(
        backend: &CudaBackend,
        tensors: &CudaTensorSet,
        spec: &DecoderLayerSpec,
        input_norm: &TensorBinding,
        bindings: StateSpaceMixerBindings<'_>,
        feed_forward: Option<NormedFeedForward>,
        hidden_size: usize,
    ) -> Result<Self> {
        let MixerSpec::StateSpace(state_space) = &spec.mixer else {
            return Err(unsupported("layer mixer is not a state-space mixer"));
        };
        let epsilon = narrow(spec.input_norm.epsilon)?;
        let geometry = geometry(state_space, epsilon)?;
        let tensor = |binding: &TensorBinding| {
            tensors
                .get(&binding.source)
                .cloned()
                .ok_or_else(|| Error::MissingTensor(binding.source.clone()))
        };
        let convolution_bias = match &bindings.convolution.storage {
            TensorStorage::Dense { bias: Some(bias), .. } => {
                Some(tensors.get(bias).cloned().ok_or_else(|| Error::MissingTensor(bias.clone()))?)
            },
            _ => None,
        };
        if convolution_bias.is_some() != state_space.convolution_bias {
            return Err(unsupported("convolution bias differs from the layer specification"));
        }
        let input_projection =
            DirectFp8CheckpointWeight::load_binding(tensors, bindings.input_projection)?;
        input_projection.validate(hidden_size, geometry.projection_width()?)?;
        let output_projection =
            DirectFp8CheckpointWeight::load_binding(tensors, bindings.output_projection)?;
        output_projection.validate(geometry.inner()?, hidden_size)?;
        Ok(Self {
            backend: backend.clone(),
            norm: LayerNormConfig {
                hidden_size,
                rms_norm_epsilon: epsilon,
                norm_weight_shift: 0.0,
            },
            geometry,
            input_norm: tensor(input_norm)?,
            input_projection,
            output_projection,
            mixer: MixerWeights {
                convolution_weight: tensor(bindings.convolution)?,
                convolution_bias,
                a_log: tensor(bindings.decay_log)?,
                skip: tensor(bindings.skip_scale)?,
                dt_bias: tensor(bindings.time_bias)?,
                norm_weight: tensor(bindings.gated_norm)?,
            },
            feed_forward,
        })
    }

    #[must_use]
    pub const fn geometry(&self) -> Mamba2Geometry {
        self.geometry
    }

    pub fn prepare_state(&self) -> Result<CudaMamba2State> {
        self.backend.prepare_mamba2_state(self.geometry)
    }

    pub fn prepare(&self, tokens: usize) -> Result<CudaMamba2HybridExecution> {
        let phase = if tokens == 1 {
            ExecutionPhase::Decode
        } else {
            ExecutionPhase::Prefill
        };
        self.prepare_phase(tokens, phase)
    }

    pub(crate) fn prepare_phase(
        &self,
        tokens: usize,
        phase: ExecutionPhase,
    ) -> Result<CudaMamba2HybridExecution> {
        CudaMamba2HybridExecution::new(self, tokens, phase)
    }
}

impl MixerWeights {
    fn view(&self) -> Result<Mamba2Parameters<'_>> {
        Ok(Mamba2Parameters {
            convolution_weight: bf16(&self.convolution_weight)?,
            convolution_bias: self.convolution_bias.as_ref().map(bf16).transpose()?,
            a_log: bf16(&self.a_log)?,
            skip: bf16(&self.skip)?,
            dt_bias: bf16(&self.dt_bias)?,
            norm_weight: bf16(&self.norm_weight)?,
        })
    }
}

fn geometry(spec: &StateSpaceSpec, norm_epsilon: f32) -> Result<Mamba2Geometry> {
    if spec.projection_bias || spec.state_dtype != StateSpaceStateDTypeSpec::Float32 {
        return Err(unsupported("Mamba-2 needs bias-free projections and an FP32 state"));
    }
    Ok(Mamba2Geometry {
        heads: spec.heads,
        head_dim: spec.head_dim,
        state_size: spec.state_size,
        groups: spec.groups,
        convolution_kernel_size: spec.convolution_kernel_size,
        time_step_minimum: narrow(spec.time_step_minimum)?,
        time_step_maximum: spec.time_step_maximum.map_or(Ok(f32::INFINITY), narrow)?,
        norm_epsilon,
    })
}

pub(super) fn bf16(tensor: &CudaTensor) -> Result<&mircuda::DeviceBuffer<mircuda::bf16>> {
    tensor.as_bf16().ok_or_else(|| Error::DTypeMismatch {
        name: tensor.name().into(),
        expected: "BF16",
    })
}

fn narrow(value: f64) -> Result<f32> {
    if !value.is_finite() || value.abs() > f64::from(f32::MAX) {
        return Err(unsupported("Mamba-2 parameter is outside finite f32 range"));
    }
    #[allow(clippy::cast_possible_truncation)]
    Ok(value as f32)
}

fn unsupported(message: &str) -> Error {
    Error::UnsupportedDecoderLayer(message.into())
}
