use mircuda::{DeviceBuffer, bf16};

use super::super::CudaBackend;
use crate::{
    CudaTensor, CudaTensorSet, Error, Result,
    backend::Bf16Linear,
    kernels::{VisionClip, VisionClipSpec, VisionElementwise, VisionElementwiseSpec},
};

#[derive(Debug)]
pub(super) struct VisionLinear {
    backend: CudaBackend,
    linear: Bf16Linear,
    weight: CudaTensor,
    bias: Option<CudaTensor>,
    clipping: Option<LinearClipping>,
    elementwise: VisionElementwise,
    rows: usize,
    input_features: usize,
    flattened_weight: bool,
}

#[derive(Debug)]
struct LinearClipping {
    input: VisionClip,
    output: VisionClip,
    input_minimum: CudaTensor,
    input_maximum: CudaTensor,
    output_minimum: CudaTensor,
    output_maximum: CudaTensor,
}

impl VisionLinear {
    pub(super) fn new(
        backend: &CudaBackend,
        tensors: &CudaTensorSet,
        prefix: &str,
        rows: usize,
        input_features: usize,
        output_features: usize,
        clipped: bool,
    ) -> Result<Self> {
        Self::new_inner(
            backend, tensors, prefix, rows, input_features, output_features, clipped, false,
        )
    }

    pub(super) fn new_flattened(
        backend: &CudaBackend,
        tensors: &CudaTensorSet,
        prefix: &str,
        rows: usize,
        input_features: usize,
        output_features: usize,
    ) -> Result<Self> {
        Self::new_inner(
            backend, tensors, prefix, rows, input_features, output_features, false, true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_inner(
        backend: &CudaBackend,
        tensors: &CudaTensorSet,
        prefix: &str,
        rows: usize,
        input_features: usize,
        output_features: usize,
        clipped: bool,
        flattened_weight: bool,
    ) -> Result<Self> {
        let (weight_name, bias_name) = if tensors.get(&format!("{prefix}.linear.weight")).is_some()
        {
            (format!("{prefix}.linear.weight"), format!("{prefix}.linear.bias"))
        } else {
            (format!("{prefix}.weight"), format!("{prefix}.bias"))
        };
        let clipping = clipped
            .then(|| {
                LinearClipping::new(backend, tensors, prefix, rows, input_features, output_features)
            })
            .transpose()?;
        Ok(Self {
            backend: backend.clone(),
            linear: backend.prepare_bf16_linear(rows, input_features, output_features)?,
            weight: required(tensors, &weight_name)?,
            bias: tensors.get(&bias_name).cloned(),
            clipping,
            elementwise: VisionElementwise::compile(
                &backend.inner.compiler,
                VisionElementwiseSpec {
                    rows,
                    columns: output_features,
                    epsilon: 0.0,
                },
            )?,
            rows,
            input_features,
            flattened_weight,
        })
    }

    pub(super) fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let stream = &self.backend.inner.stream;
        let mut clipped_input = self
            .clipping
            .as_ref()
            .map(|_| self.backend.inner.pool.allocate(stream, self.rows * self.input_features))
            .transpose()?;
        let input = if let (Some(clipping), Some(clipped_input)) =
            (self.clipping.as_ref(), clipped_input.as_mut())
        {
            clipping.input.execute(
                stream,
                input,
                &clipping.input_minimum,
                &clipping.input_maximum,
                clipped_input,
            )?;
            &*clipped_input
        } else {
            input
        };
        if self.flattened_weight {
            self.linear.execute_flattened(input, &self.weight, output)?;
        } else {
            self.linear.execute(input, &self.weight, output)?;
        }
        if let Some(bias) = self.bias.as_ref() {
            let bias = bias.as_bf16().ok_or_else(|| Error::DTypeMismatch {
                name: bias.name().into(),
                expected: "BF16",
            })?;
            let input = output.clone();
            self.elementwise.add_bias(stream, &input, bias, output)?;
        }
        if let Some(clipping) = self.clipping.as_ref() {
            let input = output.clone();
            clipping.output.execute(
                stream,
                &input,
                &clipping.output_minimum,
                &clipping.output_maximum,
                output,
            )?;
        }
        Ok(())
    }
}

impl LinearClipping {
    fn new(
        backend: &CudaBackend,
        tensors: &CudaTensorSet,
        prefix: &str,
        rows: usize,
        input_features: usize,
        output_features: usize,
    ) -> Result<Self> {
        Ok(Self {
            input: VisionClip::compile(
                &backend.inner.compiler,
                VisionClipSpec { rows, columns: input_features },
            )?,
            output: VisionClip::compile(
                &backend.inner.compiler,
                VisionClipSpec { rows, columns: output_features },
            )?,
            input_minimum: required(tensors, &format!("{prefix}.input_min"))?,
            input_maximum: required(tensors, &format!("{prefix}.input_max"))?,
            output_minimum: required(tensors, &format!("{prefix}.output_min"))?,
            output_maximum: required(tensors, &format!("{prefix}.output_max"))?,
        })
    }
}

pub(super) fn required(tensors: &CudaTensorSet, name: &str) -> Result<CudaTensor> {
    tensors.get(name).cloned().ok_or_else(|| Error::MissingTensor(name.into()))
}
