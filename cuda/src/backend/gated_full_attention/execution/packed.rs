use super::{
    AffineGatedFullAttentionConfig, CheckpointProjection, CheckpointProjectionWeight, CudaBackend,
    DenseRole, Error, ProjectionPackSplit, Result, checked,
};

pub(super) fn prepare_packed_projection(
    backend: &CudaBackend,
    config: AffineGatedFullAttentionConfig,
    tokens: usize,
    weight: Option<&CheckpointProjectionWeight>,
    query: usize,
    key_value: usize,
) -> Result<(Option<CheckpointProjection>, Option<ProjectionPackSplit>)> {
    let output = query
        .checked_add(checked(key_value, 2)?)
        .ok_or(Error::InvalidDecoderKernel("packed attention size overflow"))?;
    let projection = weight
        .map(|weight| {
            CheckpointProjection::new(
                backend,
                tokens,
                config.hidden_size,
                output,
                DenseRole::AttentionQkv,
                weight,
            )
        })
        .transpose()?;
    let split = projection
        .as_ref()
        .map(|_| {
            ProjectionPackSplit::compile(
                &backend.inner.compiler,
                tokens,
                &[query, key_value, key_value],
            )
        })
        .transpose()?;
    Ok((projection, split))
}
