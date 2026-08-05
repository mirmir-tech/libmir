use mircuda::{DeviceBuffer, PinnedBuffer, bf16};
use runtime::{backend::DecodeSequence, kv::KvStorageSpec};

use super::{
    super::{
        CudaSharedRoutedModelSession, CudaSharedRoutedModelTemplate, SharedRoutedLayerTemplate,
        boundary::{SharedRoutedEmbedding, SharedRoutedOutputHead},
    },
    layer::SharedRoutedBatchLayer,
};
use crate::{
    BatchedPagedAttentionBf16, CudaBackend, DeviceBatchSamplerBf16, Error, ExecutionPhase,
    PagedDecodeBatch, Result,
    kernels::{BatchedSplitAttentionWorkspace, ShiftedRmsNorm},
};

#[derive(Debug)]
pub(super) struct DecodeResources {
    pub(super) backend: CudaBackend,
    rows: usize,
    pub(super) token_staging: PinnedBuffer<u32>,
    token_ids: DeviceBuffer<u32>,
    position_staging: PinnedBuffer<u32>,
    positions: DeviceBuffer<u32>,
    paging: PagedDecodeBatch,
    first: DeviceBuffer<bf16>,
    second: DeviceBuffer<bf16>,
    embedding: SharedRoutedEmbedding,
    pub(super) layers: Vec<SharedRoutedBatchLayer>,
    final_norm: ShiftedRmsNorm,
    output: SharedRoutedOutputHead,
    final_norm_weight: crate::CudaTensor,
    normalized: DeviceBuffer<bf16>,
    pub(super) logits: DeviceBuffer<bf16>,
    pub(super) sampler: DeviceBatchSamplerBf16,
}

impl DecodeResources {
    pub(super) fn new(template: &CudaSharedRoutedModelTemplate, rows: usize) -> Result<Self> {
        if rows == 0 {
            return Err(Error::InvalidDecoderKernel("shared-routed decode batch is empty"));
        }
        let backend = &template.backend;
        let hidden = template.decoder.hidden_size;
        let elements = checked(rows, hidden)?;
        let allocate = |count| backend.inner.pool.allocate(&backend.inner.stream, count);
        let attention_workspace = attention_workspace(template, rows)?;
        Ok(Self {
            backend: backend.clone(),
            rows,
            token_staging: backend.inner.context.allocate_pinned(rows)?,
            token_ids: backend.inner.pool.allocate(&backend.inner.stream, rows)?,
            position_staging: backend.inner.context.allocate_pinned(checked(3, rows)?)?,
            positions: backend.inner.pool.allocate(&backend.inner.stream, checked(3, rows)?)?,
            paging: backend.prepare_paged_decode_batch(
                template.cache_spec()?,
                template.max_sequence_blocks,
                rows,
            )?,
            first: allocate(elements)?,
            second: allocate(elements)?,
            embedding: template.prepare_embedding()?,
            layers: template
                .layers
                .iter()
                .map(|layer| {
                    SharedRoutedBatchLayer::new(
                        layer,
                        rows,
                        ExecutionPhase::Decode,
                        Some(attention_workspace.clone()),
                    )
                })
                .collect::<Result<Vec<_>>>()?,
            final_norm: ShiftedRmsNorm::compile(
                &backend.inner.compiler,
                rows,
                hidden,
                template.decoder.rms_norm_eps.to_string().parse()?,
                template.norm_shift,
            )?,
            output: template.prepare_output_head(rows)?,
            final_norm_weight: template.final_norm.clone(),
            normalized: allocate(elements)?,
            logits: allocate(checked(rows, template.decoder.vocab_size)?)?,
            sampler: backend
                .prepare_device_batch_sampler_bf16(template.decoder.vocab_size, rows)?,
        })
    }

    pub(super) fn prepare(
        &mut self,
        sessions: &mut [&mut CudaSharedRoutedModelSession],
        sequences: &[DecodeSequence],
    ) -> Result<()> {
        self.validate(sessions, sequences)?;
        let tokens = sequences.iter().map(|sequence| sequence.token_id).collect::<Vec<_>>();
        let positions = sessions
            .iter()
            .map(|session| {
                u32::try_from(
                    i64::try_from(session.position())? + i64::from(session.position_delta()),
                )
                .map_err(Error::from)
            })
            .collect::<Result<Vec<_>>>()?
            .repeat(3);
        self.token_staging.copy_from_slice(&tokens)?;
        self.position_staging.copy_from_slice(&positions)?;
        let stream = &self.backend.inner.stream;
        stream.copy_to_device(&mut self.token_staging, &mut self.token_ids)?;
        stream.copy_to_device(&mut self.position_staging, &mut self.positions)?;
        let tables = sequences.iter().map(|sequence| &sequence.block_table).collect::<Vec<_>>();
        self.paging.prepare(&tables)?;
        for index in 0..self.layers.len() {
            if index.is_multiple_of(2) {
                self.layers[index]
                    .prepare(&self.first, &self.second, sessions, index, &self.paging)?;
            } else {
                self.layers[index]
                    .prepare(&self.second, &self.first, sessions, index, &self.paging)?;
            }
        }
        Ok(())
    }

    pub(super) fn execute(&mut self) -> Result<()> {
        self.embedding.execute_batch(&self.token_ids, self.rows, &mut self.first)?;
        for index in 0..self.layers.len() {
            if index.is_multiple_of(2) {
                self.layers[index].execute_prepared(
                    &self.first, &self.positions, &self.paging, &mut self.second,
                )?;
            } else {
                self.layers[index].execute_prepared(
                    &self.second, &self.positions, &self.paging, &mut self.first,
                )?;
            }
        }
        let hidden = if self.layers.len().is_multiple_of(2) {
            &self.first
        } else {
            &self.second
        };
        self.final_norm.execute(
            &self.backend.inner.stream,
            hidden,
            bf16_tensor(&self.final_norm_weight)?,
            &mut self.normalized,
        )?;
        self.output.execute(&self.normalized, &mut self.logits)
    }

    pub(super) fn capture_partitions(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| layer.capture_partitions(&self.paging))
            .max()
            .unwrap_or_default()
    }

    fn validate(
        &self,
        sessions: &[&mut CudaSharedRoutedModelSession],
        sequences: &[DecodeSequence],
    ) -> Result<()> {
        if sessions.len() != self.rows || sequences.len() != self.rows {
            return Err(Error::InvalidDecoderKernel("shared-routed decode batch row mismatch"));
        }
        for (session, sequence) in sessions.iter().zip(sequences) {
            session.embedding.validate_token(sequence.token_id)?;
            if sequence.block_table.token_len() != session.position + 1 {
                return Err(Error::InvalidPagedKv(
                    "shared-routed decode table differs from session position",
                ));
            }
        }
        Ok(())
    }
}

fn attention_workspace(
    template: &CudaSharedRoutedModelTemplate,
    rows: usize,
) -> Result<BatchedSplitAttentionWorkspace> {
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
                    rows,
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
    Ok(BatchedSplitAttentionWorkspace::new(
        backend.inner.pool.allocate(&backend.inner.stream, values)?,
        backend.inner.pool.allocate(&backend.inner.stream, statistics)?,
        backend.inner.pool.allocate(&backend.inner.stream, statistics)?,
    ))
}

pub(super) fn checked(left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right)
        .ok_or(Error::InvalidDecoderKernel("shared-routed decode batch size overflow"))
}

pub(super) fn bf16_tensor(tensor: &crate::CudaTensor) -> Result<&DeviceBuffer<bf16>> {
    tensor.as_bf16().ok_or_else(|| Error::DTypeMismatch {
        name: tensor.name().into(),
        expected: "BF16",
    })
}
