use super::{MxFp4Linear, MxFp4LinearLayout};
use crate::engine::{Array, Error, Result, Stream};

impl MxFp4Linear {
    pub(in crate::engine) fn snapshot(&self) -> Result<Self> {
        Ok(Self {
            weight: self.weight.snapshot()?,
            scales: self.scales.snapshot()?,
            bias: self.bias.snapshot()?,
            input_features: self.input_features,
            output_features: self.output_features,
            has_bias: self.has_bias,
            layout: self.layout,
        })
    }

    /// Test-only ordinary projection concatenation. Bias remains after QMM;
    /// gathered banks and unlike bias contracts are deliberately ineligible.
    pub(in crate::engine) fn join_outputs(
        &self,
        second: &Self,
        stream: &Stream,
    ) -> Result<Option<Self>> {
        if !matches!(self.layout, MxFp4LinearLayout::Matrix)
            || !matches!(second.layout, MxFp4LinearLayout::Matrix)
            || self.input_features != second.input_features
            || self.has_bias != second.has_bias
            || self.weight.dtype()? != second.weight.dtype()?
        {
            return Ok(None);
        }
        Ok(Some(Self {
            weight: Array::concatenate(&[&self.weight, &second.weight], 0, stream)?,
            scales: Array::concatenate(&[&self.scales, &second.scales], 0, stream)?,
            bias: Array::concatenate(&[&self.bias, &second.bias], 0, stream)?,
            input_features: self.input_features,
            output_features: self
                .output_features
                .checked_add(second.output_features)
                .ok_or(Error::ShapeOverflow)?,
            has_bias: self.has_bias,
            layout: MxFp4LinearLayout::Matrix,
        }))
    }
}

#[cfg(test)]
mod tests;
