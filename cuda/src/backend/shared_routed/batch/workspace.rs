//! The split-attention workspace of decode batches: sized for the widest
//! batch and shared by every row count of one model, since one decode batch
//! executes at a time and the kernels stride rows independently of the
//! batch width they were compiled for.

use runtime::kv::KvStorageSpec;

use super::super::{CudaSharedRoutedModelTemplate, SharedRoutedLayerTemplate};
use crate::{
    BatchedPagedAttentionBf16, Error, Result, backend::scratch_pool::SharedScratch,
    kernels::BatchedSplitAttentionWorkspace,
};

/// One model's decode workspace of a given size.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(in crate::backend) struct DecodeWorkspaceKey {
    model: u64,
    values: usize,
    statistics: usize,
}

/// The workspace for a batch of `rows`, allocated for the template's decode
/// width when that is wider. The handle keeps the pool entry alive.
pub(super) fn attention_workspace(
    template: &CudaSharedRoutedModelTemplate,
    rows: usize,
) -> Result<(SharedScratch<BatchedSplitAttentionWorkspace>, BatchedSplitAttentionWorkspace)> {
    let capacity = rows.max(template.decode_rows);
    let (values, statistics) = template.layers.iter().enumerate().try_fold(
        (0_usize, 0_usize),
        |(values, statistics), (layer, template_layer)| match template_layer {
            SharedRoutedLayerTemplate::Linear(_) => Ok((values, statistics)),
            SharedRoutedLayerTemplate::Full(_) => {
                let storage = KvStorageSpec::new(
                    template.cache,
                    template.decoder.layer_key_value_heads(layer),
                    template.decoder.layer_head_dim(layer),
                );
                let required = BatchedPagedAttentionBf16::workspace_lengths_for_storage(
                    &template.backend,
                    storage,
                    template.decoder.num_attention_heads,
                    template.max_sequence_blocks,
                    capacity,
                )?;
                Ok::<_, Error>((values.max(required.0), statistics.max(required.1)))
            },
        },
    )?;
    if values == 0 || statistics == 0 {
        return Err(Error::InvalidDecoderKernel(
            "shared-routed CUDA batch has no attention workspace",
        ));
    }
    let backend = &template.backend;
    let key = DecodeWorkspaceKey {
        model: template.identity,
        values,
        statistics,
    };
    let handle = backend.inner.decode_workspaces.acquire(key, || {
        Ok(BatchedSplitAttentionWorkspace::new(
            backend.inner.pool.allocate(&backend.inner.stream, values)?,
            backend.inner.pool.allocate(&backend.inner.stream, statistics)?,
            backend.inner.pool.allocate(&backend.inner.stream, statistics)?,
        ))
    })?;
    let workspace = handle.lock()?.clone();
    Ok((handle, workspace))
}
