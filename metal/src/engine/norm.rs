use super::{Array, ModelTensors, Result, Stream};

#[derive(Debug)]
pub struct NormWeight {
    weight: Array,
    formula: Formula,
}

/// How a norm scales its input before the weight.
#[derive(Debug)]
enum Formula {
    Rms,
    /// Mean-centred `LayerNorm`; MLX takes the bias these norms do not have
    /// as zeros.
    Layer {
        bias: Array,
    },
}

impl NormWeight {
    #[cfg(test)]
    pub(in crate::engine) fn from_weight(weight: Array) -> Self {
        Self { weight, formula: Formula::Rms }
    }

    pub(in crate::engine) fn load(tensors: &ModelTensors, prefix: &str) -> Result<Self> {
        Self::load_name(tensors, &format!("{prefix}.weight"))
    }

    pub(in crate::engine) fn load_name(tensors: &ModelTensors, weight: &str) -> Result<Self> {
        Ok(Self {
            weight: tensors.get(weight)?,
            formula: Formula::Rms,
        })
    }

    pub(in crate::engine) fn load_shifted(
        tensors: &ModelTensors,
        prefix: &str,
        shift: f32,
        stream: &Stream,
    ) -> Result<Self> {
        let weight = tensors.get(&format!("{prefix}.weight"))?.add_scalar(shift, stream)?;
        Ok(Self { weight, formula: Formula::Rms })
    }

    pub(in crate::engine) fn load_adjusted(
        tensors: &ModelTensors,
        prefix: &str,
        shift: f32,
        stream: &Stream,
    ) -> Result<Self> {
        if shift == 0.0 {
            return Self::load(tensors, prefix);
        }
        Self::load_shifted(tensors, prefix, shift, stream)
    }

    pub(in crate::engine) fn load_name_adjusted(
        tensors: &ModelTensors,
        weight: &str,
        shift: f32,
        stream: &Stream,
    ) -> Result<Self> {
        let weight = tensors.get(weight)?;
        let weight = if shift == 0.0 {
            weight
        } else {
            weight.add_scalar(shift, stream)?
        };
        Ok(Self { weight, formula: Formula::Rms })
    }

    pub(in crate::engine) fn load_optional(
        tensors: &ModelTensors,
        prefix: &str,
    ) -> Result<Option<Self>> {
        tensors
            .get_optional(&format!("{prefix}.weight"))
            .map(|weight| weight.map(|weight| Self { weight, formula: Formula::Rms }))
    }

    pub(in crate::engine) fn apply(
        &self,
        input: &Array,
        eps: f32,
        stream: &Stream,
    ) -> Result<Array> {
        match &self.formula {
            Formula::Rms => input.rms_norm(&self.weight, eps, stream),
            Formula::Layer { bias } => Array::from_native(stream.native().graph().layer_norm(
                input.native(),
                self.weight.native(),
                bias.native(),
                eps,
            )?),
        }
    }

    /// The same weight in a mean-centred `LayerNorm` without bias (Cohere).
    /// A `LayerNorm` with this weight and `bias`, zero when absent.
    pub(in crate::engine) fn into_layer(
        self,
        bias: Option<Array>,
        stream: &Stream,
    ) -> Result<Self> {
        let bias = match bias {
            Some(bias) => bias,
            None => self.weight.multiply_scalar(0.0, stream)?,
        };
        Ok(Self {
            weight: self.weight,
            formula: Formula::Layer { bias },
        })
    }

    pub(super) fn native_clone(&self) -> mirtal::Array {
        self.weight.native().clone()
    }
}
