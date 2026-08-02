use crate::{CudaBackend, CudaTensor, Error, Result, kernels::DenseExpertCanonicalizer};

pub(super) fn canonicalize(
    backend: &CudaBackend,
    operation: &DenseExpertCanonicalizer,
    source: &CudaTensor,
    experts: usize,
    input: usize,
    output: usize,
) -> Result<CudaTensor> {
    let source_buffer = source.as_bf16().ok_or_else(|| Error::DTypeMismatch {
        name: source.name().into(),
        expected: "BF16",
    })?;
    let mut buffer = backend.inner.pool.allocate(&backend.inner.stream, source_buffer.len())?;
    operation.execute(&backend.inner.stream, source_buffer, &mut buffer, experts, input, output)?;
    Ok(CudaTensor::from_bf16(
        format!("{}#canonical", source.name()),
        vec![experts, output, input],
        buffer,
    ))
}
