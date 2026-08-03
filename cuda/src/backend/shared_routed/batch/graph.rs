use mircuda::{DeviceBuffer, PinnedBuffer, bf16};
use runtime::backend::DecodeSequence;

use super::{
    super::{
        CudaSharedRoutedModelSession, CudaSharedRoutedModelTemplate,
        boundary::{SharedRoutedEmbedding, SharedRoutedOutputHead},
    },
    layer::SharedRoutedBatchLayer,
};
use crate::{CudaBackend, Error, Result, kernels::ShiftedRmsNorm};

#[derive(Debug)]
pub(super) struct DecodeResources {
    backend: CudaBackend,
    rows: usize,
    max_blocks: usize,
    token_staging: PinnedBuffer<u32>,
    token_ids: DeviceBuffer<u32>,
    position_staging: PinnedBuffer<u32>,
    positions: DeviceBuffer<u32>,
    first: DeviceBuffer<bf16>,
    second: DeviceBuffer<bf16>,
    embedding: SharedRoutedEmbedding,
    layers: Vec<SharedRoutedBatchLayer>,
    final_norm: ShiftedRmsNorm,
    output: SharedRoutedOutputHead,
    final_norm_weight: crate::CudaTensor,
    normalized: DeviceBuffer<bf16>,
    logits: DeviceBuffer<bf16>,
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
        Ok(Self {
            backend: backend.clone(),
            rows,
            max_blocks: template.max_sequence_blocks,
            token_staging: backend.inner.context.allocate_pinned(rows)?,
            token_ids: backend.inner.pool.allocate(&backend.inner.stream, rows)?,
            position_staging: backend.inner.context.allocate_pinned(checked(3, rows)?)?,
            positions: backend.inner.pool.allocate(&backend.inner.stream, checked(3, rows)?)?,
            first: allocate(elements)?,
            second: allocate(elements)?,
            embedding: template.prepare_embedding()?,
            layers: template
                .layers
                .iter()
                .map(|layer| SharedRoutedBatchLayer::new(layer, rows))
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
        for index in 0..self.layers.len() {
            if index.is_multiple_of(2) {
                self.layers[index].prepare(
                    &self.first, &self.second, sessions, sequences, index, self.max_blocks,
                )?;
            } else {
                self.layers[index].prepare(
                    &self.second, &self.first, sessions, sequences, index, self.max_blocks,
                )?;
            }
        }
        Ok(())
    }

    pub(super) fn execute(&mut self) -> Result<()> {
        self.embedding.execute_batch(&self.token_ids, self.rows, &mut self.first)?;
        for index in 0..self.layers.len() {
            if index.is_multiple_of(2) {
                self.layers[index]
                    .execute_prepared(&self.first, &self.positions, &mut self.second)?;
            } else {
                self.layers[index]
                    .execute_prepared(&self.second, &self.positions, &mut self.first)?;
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

    pub(super) fn commit(
        &mut self,
        sessions: &mut [&mut CudaSharedRoutedModelSession],
    ) -> Result<()> {
        for index in 0..self.layers.len() {
            self.layers[index].commit(sessions, index)?;
        }
        for (row, session) in sessions.iter_mut().enumerate() {
            let start = checked(row, session.logits.len())?;
            self.backend.inner.stream.copy_device_range(
                &self.logits,
                start..start + session.logits.len(),
                &mut session.logits,
                0,
            )?;
            session.position = session
                .position
                .checked_add(1)
                .ok_or(Error::InvalidDecoderKernel("shared-routed session position overflow"))?;
        }
        Ok(())
    }

    pub(super) fn capture_partitions(&self) -> usize {
        self.layers
            .iter()
            .map(SharedRoutedBatchLayer::capture_partitions)
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
