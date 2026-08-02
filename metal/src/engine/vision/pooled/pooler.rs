use models::weights::TensorBinding;

use super::dimension;
use crate::engine::{Array, Dtype, Error, ModelTensors, Result, Stream, binding::BoundLinear};

#[derive(Debug)]
pub(super) struct VisionPooler {
    bias: Option<Array>,
    scale: Option<Array>,
    projection: BoundLinear,
    hidden_size: usize,
    kernel: usize,
    eps: f32,
}

impl VisionPooler {
    pub(super) fn load(
        tensors: &ModelTensors,
        hidden_size: usize,
        kernel: usize,
        standardize: bool,
        eps: f32,
        projection: &TensorBinding,
        stream: &Stream,
    ) -> Result<Self> {
        let (bias, scale) = if standardize {
            (
                Some(tensors.get("model.vision_tower.std_bias")?),
                Some(tensors.get("model.vision_tower.std_scale")?),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            bias,
            scale,
            projection: BoundLinear::load(tensors, projection, stream)?,
            hidden_size,
            kernel,
            eps,
        })
    }

    pub(super) fn forward(
        &self,
        input: &Array,
        grid_height: usize,
        grid_width: usize,
        stream: &Stream,
    ) -> Result<Array> {
        let shape = input.shape()?;
        validate_grid(&shape, grid_height, grid_width, self.hidden_size, self.kernel)?;
        let batch = shape[0];
        let output_height = grid_height / self.kernel;
        let output_width = grid_width / self.kernel;
        let pooled_tokens = output_height * output_width;
        let kernel = dimension(self.kernel, "pooling kernel")?;
        let hidden = dimension(self.hidden_size, "hidden size")?;
        let pooled = input
            .astype(Dtype::Float32, stream)?
            .reshape(
                &[
                    batch,
                    dimension(output_height, "pooled height")?,
                    kernel,
                    dimension(output_width, "pooled width")?,
                    kernel,
                    hidden,
                ],
                stream,
            )?
            .reduce_sum(4, false, stream)?
            .reduce_sum(2, false, stream)?
            .multiply_scalar(1.0 / (self.kernel * self.kernel).to_string().parse::<f32>()?, stream)?
            .reshape(&[batch, dimension(pooled_tokens, "pooled token count")?, hidden], stream)?
            .astype_like(input, stream)?
            .astype(Dtype::Float32, stream)?
            .multiply_scalar(self.hidden_size.to_string().parse::<f32>()?.sqrt(), stream)?;
        let standardized = match (&self.bias, &self.scale) {
            (Some(bias), Some(scale)) => pooled
                .add(&bias.multiply_scalar(-1.0, stream)?, stream)?
                .multiply(scale, stream)?,
            (None, None) => pooled,
            _ => {
                return Err(Error::InvalidModel("incomplete pooled vision standardization".into()));
            },
        };
        self.projection.forward(
            &standardized.astype_like(input, stream)?.rms_norm_unit(self.eps, stream)?,
            stream,
        )
    }

    #[cfg(test)]
    pub(super) fn from_projection(
        projection_weight: &Array,
        hidden_size: usize,
        kernel: usize,
        eps: f32,
        stream: &Stream,
    ) -> Result<Self> {
        Ok(Self {
            bias: None,
            scale: None,
            projection: BoundLinear::Dense(crate::engine::DenseLinear::from_arrays(
                projection_weight,
                None,
                stream,
            )?),
            hidden_size,
            kernel,
            eps,
        })
    }
}

fn validate_grid(
    shape: &[i32],
    height: usize,
    width: usize,
    hidden: usize,
    kernel: usize,
) -> Result<()> {
    let sequence = height.checked_mul(width).ok_or(Error::ShapeOverflow)?;
    if shape.len() == 3
        && usize::try_from(shape[1])? == sequence
        && usize::try_from(shape[2])? == hidden
        && height.is_multiple_of(kernel)
        && width.is_multiple_of(kernel)
    {
        return Ok(());
    }
    Err(Error::InvalidModel(format!(
        "pooled vision hidden shape {shape:?} is incompatible with grid {height}x{width}, hidden {hidden}, kernel {kernel}"
    )))
}
