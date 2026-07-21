use super::{Array, ModelTensors, Result, Stream};

#[derive(Debug)]
pub struct NormWeight {
    weight: Array,
}

impl NormWeight {
    #[cfg(test)]
    pub(in crate::engine) fn from_weight(weight: Array) -> Self {
        Self { weight }
    }

    pub(in crate::engine) fn load(tensors: &ModelTensors, prefix: &str) -> Result<Self> {
        Self::load_name(tensors, &format!("{prefix}.weight"))
    }

    pub(in crate::engine) fn load_name(tensors: &ModelTensors, weight: &str) -> Result<Self> {
        Ok(Self { weight: tensors.get(weight)? })
    }

    pub(in crate::engine) fn load_shifted(
        tensors: &ModelTensors,
        prefix: &str,
        shift: f32,
        stream: &Stream,
    ) -> Result<Self> {
        let weight = tensors.get(&format!("{prefix}.weight"))?.add_scalar(shift, stream)?;
        Ok(Self { weight })
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
        Ok(Self { weight })
    }

    pub(in crate::engine) fn load_optional(
        tensors: &ModelTensors,
        prefix: &str,
    ) -> Result<Option<Self>> {
        tensors
            .get_optional(&format!("{prefix}.weight"))
            .map(|weight| weight.map(|weight| Self { weight }))
    }

    pub(in crate::engine) fn apply(
        &self,
        input: &Array,
        eps: f32,
        stream: &Stream,
    ) -> Result<Array> {
        input.rms_norm(&self.weight, eps, stream)
    }

    pub(super) fn native_clone(&self) -> mirtal::Array {
        self.weight.native().clone()
    }
}
