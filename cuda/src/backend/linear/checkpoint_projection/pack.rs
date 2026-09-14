use super::CheckpointProjectionWeight;
use crate::{CudaBackend, CudaTensor, Error, Result};

impl CheckpointProjectionWeight {
    /// Concatenates output rows and replaces the sources with views of the same
    /// allocation, so retaining the original projections does not retain
    /// copies.
    pub(in crate::backend) fn pack_dense_pair(
        backend: &CudaBackend,
        first: &mut Self,
        second: &mut Self,
    ) -> Result<Option<Self>> {
        let (Some(left), Some(right)) = (first.dense_bf16(), second.dense_bf16()) else {
            return Ok(None);
        };
        let ([left_rows, columns], [right_rows, right_columns]) = (left.shape(), right.shape())
        else {
            return Ok(None);
        };
        if columns != right_columns || *columns == 0 || *left_rows == 0 || *right_rows == 0 {
            return Ok(None);
        }
        let rows = left_rows
            .checked_add(*right_rows)
            .ok_or(Error::InvalidExecutionPlan("packed dense rows overflow"))?;
        let elements = rows
            .checked_mul(*columns)
            .ok_or(Error::InvalidExecutionPlan("packed dense size overflow"))?;
        let left_buffer =
            left.as_bf16().ok_or(Error::InvalidExecutionPlan("missing BF16 weight"))?;
        let right_buffer =
            right.as_bf16().ok_or(Error::InvalidExecutionPlan("missing BF16 weight"))?;
        let split = left_rows * columns;
        if left_buffer.len() != split || right_buffer.len() != elements - split {
            return Err(Error::InvalidExecutionPlan("packed dense storage shape mismatch"));
        }
        let stream = &backend.inner.stream;
        let mut packed = backend.inner.pool.allocate(stream, elements)?;
        stream.copy_device_range(left_buffer, 0..split, &mut packed, 0)?;
        stream.copy_device_range(right_buffer, 0..right_buffer.len(), &mut packed, split)?;
        let left_view = CudaTensor::from_bf16(
            left.name().into(),
            left.shape().to_vec(),
            packed.slice(0..split)?,
        );
        let right_view = CudaTensor::from_bf16(
            right.name().into(),
            right.shape().to_vec(),
            packed.slice(split..elements)?,
        );
        let weight = CudaTensor::from_bf16(
            format!("{}+{}", left.name(), right.name()),
            vec![rows, *columns],
            packed,
        );
        *first = Self::Dense(left_view);
        *second = Self::Dense(right_view);
        Ok(Some(Self::Dense(weight)))
    }

    pub(in crate::backend) fn pack_direct_fp8<const N: usize>(
        backend: &CudaBackend,
        weights: [&Self; N],
    ) -> Result<Option<Self>> {
        let direct = weights.map(|weight| match weight {
            Self::DirectFp8(weight) => Some(weight),
            _ => None,
        });
        let Some(direct) = direct.into_iter().collect::<Option<Vec<_>>>() else {
            return Ok(None);
        };
        let Ok(direct): std::result::Result<[&crate::DirectFp8CheckpointWeight; N], _> =
            direct.try_into()
        else {
            return Ok(None);
        };
        Ok(crate::DirectFp8CheckpointWeight::pack::<N>(backend, direct)?.map(Self::DirectFp8))
    }
}
