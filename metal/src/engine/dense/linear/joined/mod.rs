use super::*;

impl DenseLinear {
    pub(in crate::engine) fn join_outputs(
        &self,
        second: &Self,
        stream: &Stream,
    ) -> Result<Option<Self>> {
        let a = self.transposed_weight.shape()?;
        let b = second.transposed_weight.shape()?;
        if self.clipping.is_some()
            || second.clipping.is_some()
            || a.len() != 2
            || b.len() != 2
            || a[0] != b[0]
            || self.transposed_weight.dtype()? != second.transposed_weight.dtype()?
        {
            return Ok(None);
        }
        let bias = match (&self.bias, &second.bias) {
            (None, None) => None,
            (Some(first), Some(second))
                if first.shape()? == [a[1]]
                    && second.shape()? == [b[1]]
                    && first.dtype()? == second.dtype()? =>
            {
                Some(Array::concatenate(&[first, second], 0, stream)?)
            },
            _ => return Ok(None),
        };
        Ok(Some(Self {
            // Join output-major matrices and restore the transposed view.
            // Concatenating KxN directly would turn the checkpoint's column
            // layout into a row-major allocation and change native dispatch.
            transposed_weight: Array::concatenate(
                &[
                    &self.transposed_weight.transpose(&[1, 0], stream)?,
                    &second.transposed_weight.transpose(&[1, 0], stream)?,
                ],
                0,
                stream,
            )?
            .transpose(&[1, 0], stream)?,
            bias,
            clipping: None,
        }))
    }

    pub(in crate::engine) fn snapshot_unclipped(&self) -> Result<Self> {
        if self.clipping.is_some() {
            return Err(Error::InvalidModel(
                "cannot snapshot clipped projection for K/V fusion".into(),
            ));
        }
        Ok(Self {
            transposed_weight: self.transposed_weight.snapshot()?,
            bias: self.bias.as_ref().map(Array::snapshot).transpose()?,
            clipping: None,
        })
    }

    pub(in crate::engine) fn joined_probe_layout(&self) -> Result<(usize, usize)> {
        let shape = self.transposed_weight.shape()?;
        if shape.len() != 2 {
            return Err(Error::InvalidModel("joined probe requires a matrix".into()));
        }
        let bias = self.bias.as_ref().map(Array::byte_len).transpose()?.unwrap_or_default();
        Ok((
            usize::try_from(shape[1])?,
            self.transposed_weight
                .byte_len()?
                .checked_add(bias)
                .ok_or(Error::ShapeOverflow)?,
        ))
    }
}

#[cfg(test)]
mod tests;
