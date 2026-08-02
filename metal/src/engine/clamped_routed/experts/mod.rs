use models::weights::{
    BindingTransform, RoutedDecoderLayerBindings, RoutedExpertBindings, TensorBinding,
    TensorStorage,
};

use super::{
    config::ClampedRoutedConfig,
    dense::{dense_experts, fused_dense_experts},
    projection::BoundLinear,
};
use crate::engine::{Array, ModelTensors, Result, RouterOutput, Stream, kernels::MxFp4Shape};

mod native;
use native::native_layout;

#[derive(Debug)]
pub(super) struct ClampedRoutedExperts {
    router: BoundLinear,
    layout: ExpertLayout,
    limit: Array,
    config: ClampedRoutedConfig,
}

#[derive(Debug)]
enum ExpertLayout {
    Dense {
        gate: BoundLinear,
        up: BoundLinear,
        down: BoundLinear,
    },
    DenseFused {
        gate_up: BoundLinear,
        down: BoundLinear,
        interleaved: bool,
    },
    Native {
        gate_up_blocks: Array,
        gate_up_scales: Array,
        gate_up_bias: Array,
        down_blocks: Array,
        down_scales: Array,
        down_bias: Array,
    },
    Mlx {
        gate_blocks: Array,
        gate_scales: Array,
        gate_bias: Array,
        up_blocks: Array,
        up_scales: Array,
        up_bias: Array,
        down_blocks: Array,
        down_scales: Array,
        down_bias: Array,
    },
}

impl ClampedRoutedExperts {
    pub fn load(
        tensors: &ModelTensors,
        bindings: RoutedDecoderLayerBindings<'_>,
        config: ClampedRoutedConfig,
        stream: &Stream,
    ) -> Result<Self> {
        let layout = match bindings.experts {
            RoutedExpertBindings::InterleavedGateUp { gate_up, down } => {
                if matches!(&gate_up.storage, TensorStorage::Dense { .. })
                    && matches!(&down.storage, TensorStorage::Dense { .. })
                {
                    ExpertLayout::DenseFused {
                        gate_up: BoundLinear::load(tensors, gate_up, stream)?,
                        down: BoundLinear::load(tensors, down, stream)?,
                        interleaved: gate_up
                            .transforms
                            .contains(&BindingTransform::FusedGateUp { interleaved: true }),
                    }
                } else {
                    native_layout(tensors, gate_up, down, config)?
                }
            },
            RoutedExpertBindings::SeparateGateUp { gate, up, down } => {
                if matches!(&gate.storage, TensorStorage::Dense { .. })
                    && matches!(&up.storage, TensorStorage::Dense { .. })
                    && matches!(&down.storage, TensorStorage::Dense { .. })
                {
                    ExpertLayout::Dense {
                        gate: BoundLinear::load(tensors, gate, stream)?,
                        up: BoundLinear::load(tensors, up, stream)?,
                        down: BoundLinear::load(tensors, down, stream)?,
                    }
                } else {
                    mlx_layout(tensors, gate, up, down)?
                }
            },
            RoutedExpertBindings::Individual { .. } => {
                return Err(crate::engine::Error::InvalidModel(
                    "clamped routed Metal execution requires stacked expert tensors".into(),
                ));
            },
        };
        Ok(Self {
            router: BoundLinear::load(tensors, bindings.router, stream)?,
            layout,
            limit: Array::from_f32(&[config.swiglu_limit], &[])?,
            config,
        })
    }

    pub fn forward(&self, input: &Array, stream: &Stream) -> Result<Array> {
        let input_shape = input.shape()?;
        let tokens =
            input_shape[..input_shape.len() - 1].iter().try_fold(1_usize, |total, value| {
                total
                    .checked_mul(usize::try_from(*value)?)
                    .ok_or(crate::engine::Error::ShapeOverflow)
            })?;
        let flat = input.reshape(&[i32::try_from(tokens)?, self.config.hidden], stream)?;
        let scores = self.router.forward(&flat, stream)?;
        let routing = scores.router_top_k_unit(self.config.top_k, stream)?;
        let output = self.execute(&flat, &routing, tokens, stream)?;
        output.reshape(&input_shape, stream)
    }

    fn execute(
        &self,
        input: &Array,
        routing: &RouterOutput,
        tokens: usize,
        stream: &Stream,
    ) -> Result<Array> {
        let shape = MxFp4Shape {
            tokens,
            top_k: usize::try_from(self.config.top_k)?,
            hidden: usize::try_from(self.config.hidden)?,
            intermediate: usize::try_from(self.config.intermediate)?,
        };
        match &self.layout {
            ExpertLayout::Dense { gate, up, down } => {
                dense_experts(input, routing, [gate, up, down], &self.limit, shape, stream)
            },
            ExpertLayout::DenseFused { gate_up, down, interleaved } => fused_dense_experts(
                input, routing, gate_up, down, *interleaved, &self.limit, shape, stream,
            ),
            ExpertLayout::Native {
                gate_up_blocks,
                gate_up_scales,
                gate_up_bias,
                down_blocks,
                down_scales,
                down_bias,
            } => {
                let activated = stream.mxfp4_gate_up(
                    [
                        input, gate_up_blocks, gate_up_scales, gate_up_bias, &routing.indices,
                        &self.limit,
                    ],
                    shape,
                )?;
                stream.mxfp4_down(
                    [
                        &activated, down_blocks, down_scales, down_bias, &routing.indices,
                        &routing.weights,
                    ],
                    shape,
                )
            },
            ExpertLayout::Mlx {
                gate_blocks,
                gate_scales,
                gate_bias,
                up_blocks,
                up_scales,
                up_bias,
                down_blocks,
                down_scales,
                down_bias,
            } => {
                let activated = stream.mxfp4_split_gate_up(
                    [
                        input, gate_blocks, gate_scales, gate_bias, up_blocks, up_scales, up_bias,
                        &routing.indices, &self.limit,
                    ],
                    shape,
                )?;
                stream.mxfp4_u32_down(
                    [
                        &activated, down_blocks, down_scales, down_bias, &routing.indices,
                        &routing.weights,
                    ],
                    shape,
                )
            },
        }
    }
}

fn mlx_layout(
    tensors: &ModelTensors,
    gate: &TensorBinding,
    up: &TensorBinding,
    down: &TensorBinding,
) -> Result<ExpertLayout> {
    let (gate_scales, gate_bias) = affine_expert_companions(gate)?;
    let (up_scales, up_bias) = affine_expert_companions(up)?;
    let (down_scales, down_bias) = affine_expert_companions(down)?;
    Ok(ExpertLayout::Mlx {
        gate_blocks: tensors.get(&gate.source)?,
        gate_scales: tensors.get(gate_scales)?,
        gate_bias: tensors.get(gate_bias)?,
        up_blocks: tensors.get(&up.source)?,
        up_scales: tensors.get(up_scales)?,
        up_bias: tensors.get(up_bias)?,
        down_blocks: tensors.get(&down.source)?,
        down_scales: tensors.get(down_scales)?,
        down_bias: tensors.get(down_bias)?,
    })
}

fn affine_expert_companions(binding: &TensorBinding) -> Result<(&str, &str)> {
    let TensorStorage::AffineQuantized { scales, output_bias: Some(bias), .. } = &binding.storage
    else {
        return Err(crate::engine::Error::InvalidQuantization(format!(
            "clamped-routed affine expert binding lacks companions: {}",
            binding.source
        )));
    };
    Ok((scales, bias))
}
