use crate::engine::{Array, Error, ModelTensors, Result, Stream};

#[derive(Debug)]
pub struct DenseLinear {
    transposed_weight: Array,
    bias: Option<Array>,
    clipping: Option<LinearClipping>,
}

#[derive(Debug)]
struct LinearClipping {
    input_minimum: Array,
    input_maximum: Array,
    output_minimum: Array,
    output_maximum: Array,
}

impl DenseLinear {
    pub fn load(tensors: &ModelTensors, prefix: &str, stream: &Stream) -> Result<Self> {
        Self::load_names(
            tensors,
            &format!("{prefix}.weight"),
            Some(&format!("{prefix}.bias")),
            None,
            stream,
        )
    }

    pub fn load_clippable(tensors: &ModelTensors, prefix: &str, stream: &Stream) -> Result<Self> {
        let linear = format!("{prefix}.linear");
        if tensors.contains(&format!("{linear}.weight"))? {
            Self::load_names(
                tensors,
                &format!("{linear}.weight"),
                Some(&format!("{linear}.bias")),
                Some(prefix),
                stream,
            )
        } else {
            Self::load_names(
                tensors,
                &format!("{prefix}.weight"),
                Some(&format!("{prefix}.bias")),
                Some(prefix),
                stream,
            )
        }
    }

    pub fn load_flattened(tensors: &ModelTensors, prefix: &str, stream: &Stream) -> Result<Self> {
        let weight = tensors.get(&format!("{prefix}.weight"))?;
        let shape = weight.shape()?;
        let output = *shape.first().ok_or(Error::ShapeOverflow)?;
        let input = shape[1..].iter().try_fold(1_i32, |total, dimension| {
            total.checked_mul(*dimension).ok_or(Error::ShapeOverflow)
        })?;
        let weight = weight.reshape(&[output, input], stream)?;
        Ok(Self {
            transposed_weight: weight.transpose(&[1, 0], stream)?,
            bias: tensors.get_optional(&format!("{prefix}.bias"))?,
            clipping: None,
        })
    }

    pub(in crate::engine) fn load_names(
        tensors: &ModelTensors,
        weight: &str,
        bias: Option<&str>,
        clipping_prefix: Option<&str>,
        stream: &Stream,
    ) -> Result<Self> {
        let weight = tensors.get(weight)?;
        let transposed_weight = weight.transpose(&[1, 0], stream)?;
        let bias = bias.map(|name| tensors.get_optional(name)).transpose()?.flatten();
        let clipping = clipping_prefix
            .map(|prefix| LinearClipping::load_optional(tensors, prefix))
            .transpose()?
            .flatten();
        Ok(Self { transposed_weight, bias, clipping })
    }

    pub fn forward(&self, input: &Array, stream: &Stream) -> Result<Array> {
        let clipped_input = self
            .clipping
            .as_ref()
            .map(|clipping| input.clip(&clipping.input_minimum, &clipping.input_maximum, stream))
            .transpose()?;
        let input = clipped_input.as_ref().unwrap_or(input);
        let output = input.matmul(&self.transposed_weight, stream)?;
        let output = match self.bias.as_ref() {
            Some(bias) => output.add(bias, stream),
            None => Ok(output),
        }?;
        match self.clipping.as_ref() {
            Some(clipping) => {
                output.clip(&clipping.output_minimum, &clipping.output_maximum, stream)
            },
            None => Ok(output),
        }
    }

    #[cfg(test)]
    pub(in crate::engine) fn from_arrays(
        weight: &Array,
        bias: Option<Array>,
        stream: &Stream,
    ) -> Result<Self> {
        Ok(Self {
            transposed_weight: weight.transpose(&[1, 0], stream)?,
            bias,
            clipping: None,
        })
    }

    #[cfg(test)]
    pub(in crate::engine) fn from_clipped_arrays(
        weight: &Array,
        input_bounds: (Array, Array),
        output_bounds: (Array, Array),
        stream: &Stream,
    ) -> Result<Self> {
        Ok(Self {
            transposed_weight: weight.transpose(&[1, 0], stream)?,
            bias: None,
            clipping: Some(LinearClipping {
                input_minimum: input_bounds.0,
                input_maximum: input_bounds.1,
                output_minimum: output_bounds.0,
                output_maximum: output_bounds.1,
            }),
        })
    }
}

impl LinearClipping {
    fn load_optional(tensors: &ModelTensors, prefix: &str) -> Result<Option<Self>> {
        let Some(input_minimum) = tensors.get_optional(&format!("{prefix}.input_min"))? else {
            return Ok(None);
        };
        Ok(Some(Self {
            input_minimum,
            input_maximum: tensors.get(&format!("{prefix}.input_max"))?,
            output_minimum: tensors.get(&format!("{prefix}.output_min"))?,
            output_maximum: tensors.get(&format!("{prefix}.output_max"))?,
        }))
    }
}
