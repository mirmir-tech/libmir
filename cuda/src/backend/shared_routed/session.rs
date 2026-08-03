use mircuda::{DeviceBuffer, bf16};
use runtime::{backend::SamplingLogits, kv::BlockTable};
use uuid::Uuid;

use super::{
    CudaSharedRoutedLayerState, CudaSharedRoutedModelTemplate,
    boundary::{SharedRoutedEmbedding, SharedRoutedOutputHead},
    checkpoint::SharedRoutedCheckpoint,
    plan::SharedRoutedExecutionPlan,
    position::text_positions,
};
use crate::{
    CudaBackend, DeviceSamplerBf16, Error, Result,
    kernels::{SelectRowBf16, ShiftedRmsNorm},
};

/// One mutable CUDA session for a shared-routed mixed-mixer model.
pub struct CudaSharedRoutedModelSession {
    template: CudaSharedRoutedModelTemplate,
    pub(super) embedding: SharedRoutedEmbedding,
    pub(super) states: Vec<CudaSharedRoutedLayerState>,
    final_norm: ShiftedRmsNorm,
    output: SharedRoutedOutputHead,
    sampler: DeviceSamplerBf16,
    select_row: SelectRowBf16,
    last_hidden: DeviceBuffer<bf16>,
    normalized: DeviceBuffer<bf16>,
    pub(super) logits: DeviceBuffer<bf16>,
    pub(super) position: usize,
    position_delta: i32,
}

impl CudaSharedRoutedModelSession {
    pub(super) fn new(
        template: &CudaSharedRoutedModelTemplate,
        caches: &[Option<crate::PagedKvCache>],
    ) -> Result<Self> {
        let hidden = template.decoder.hidden_size;
        let vocab = template.decoder.vocab_size;
        let epsilon = template.decoder.rms_norm_eps.to_string().parse()?;
        let backend = &template.backend;
        Ok(Self {
            template: template.clone(),
            embedding: template.prepare_embedding()?,
            states: template.prepare_states(caches)?,
            final_norm: ShiftedRmsNorm::compile(
                &backend.inner.compiler,
                1,
                hidden,
                epsilon,
                template.norm_shift,
            )?,
            output: template.prepare_output_head(1)?,
            sampler: backend.prepare_device_sampler_bf16(vocab)?,
            select_row: SelectRowBf16::compile(&backend.inner.compiler, hidden)?,
            last_hidden: allocate(backend, hidden)?,
            normalized: allocate(backend, hidden)?,
            logits: allocate(backend, vocab)?,
            position: 0,
            position_delta: 0,
        })
    }

    pub fn prefill(
        &mut self,
        session_id: Uuid,
        tokens: &[u32],
        table: &BlockTable,
    ) -> Result<&DeviceBuffer<bf16>> {
        let positions = text_positions(self.position, tokens.len(), self.position_delta)?;
        self.prefill_inner(session_id, tokens, &positions, table, None, None)
    }

    pub fn prefill_with_positions(
        &mut self,
        session_id: Uuid,
        tokens: &[u32],
        positions: &[u32],
        table: &BlockTable,
        image_span: Option<(usize, usize)>,
    ) -> Result<&DeviceBuffer<bf16>> {
        self.prefill_inner(session_id, tokens, positions, table, image_span, None)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn prefill_vision(
        &mut self,
        session_id: Uuid,
        tokens: &[u32],
        positions: &[u32],
        table: &BlockTable,
        image_span: (usize, usize),
        image: &DeviceBuffer<bf16>,
        position_delta: i32,
    ) -> Result<&DeviceBuffer<bf16>> {
        self.prefill_inner(session_id, tokens, positions, table, Some(image_span), Some(image))?;
        self.position_delta = position_delta;
        Ok(&self.logits)
    }

    fn prefill_inner(
        &mut self,
        session_id: Uuid,
        tokens: &[u32],
        positions: &[u32],
        table: &BlockTable,
        image_span: Option<(usize, usize)>,
        image: Option<&DeviceBuffer<bf16>>,
    ) -> Result<&DeviceBuffer<bf16>> {
        let count = tokens.len();
        let plans = self.template.plans.clone();
        let mut plans = plans
            .lock()
            .map_err(|_| Error::State("shared-routed execution plan cache is poisoned".into()))?;
        let plan = match plans.entry(count) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(SharedRoutedExecutionPlan::new(&self.template, count)?)
            },
        };
        self.execute_plan(session_id, tokens, positions, table, image_span, image, plan)?;
        drop(plans);
        Ok(&self.logits)
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_plan(
        &mut self,
        session_id: Uuid,
        tokens: &[u32],
        positions: &[u32],
        table: &BlockTable,
        image_span: Option<(usize, usize)>,
        image: Option<&DeviceBuffer<bf16>>,
        plan: &mut SharedRoutedExecutionPlan,
    ) -> Result<()> {
        self.validate(tokens, positions, table)?;
        for token in tokens {
            self.embedding.validate_token(*token)?;
        }
        let count = tokens.len();
        plan.upload(&self.template, tokens, positions)?;
        let hidden = plan.execute(
            &self.template, &self.embedding, &mut self.states, session_id, table, self.position,
            image_span, image,
        )?;
        self.select_row.execute(
            &self.template.backend.inner.stream,
            hidden,
            &mut self.last_hidden,
            count - 1,
            count,
        )?;
        self.final_norm.execute(
            &self.template.backend.inner.stream,
            &self.last_hidden,
            bf16_tensor(self.template.final_norm_weight())?,
            &mut self.normalized,
        )?;
        self.output.execute(&self.normalized, &mut self.logits)?;
        self.position = self
            .position
            .checked_add(count)
            .ok_or(Error::InvalidDecoderKernel("shared-routed session position overflow"))?;
        Ok(())
    }

    pub fn decode(
        &mut self,
        session_id: Uuid,
        token: u32,
        table: &BlockTable,
    ) -> Result<&DeviceBuffer<bf16>> {
        self.prefill(session_id, &[token], table)
    }

    pub fn sample(&mut self, policy: SamplingLogits) -> Result<&DeviceBuffer<u32>> {
        self.sampler.sample(&self.logits, policy)
    }

    pub(crate) fn checkpoint(&self) -> Result<SharedRoutedCheckpoint> {
        SharedRoutedCheckpoint::capture(&self.states, self.position, self.position_delta)
    }

    pub(crate) fn restore_checkpoint(&mut self, checkpoint: &SharedRoutedCheckpoint) -> Result<()> {
        checkpoint.restore(&mut self.states)?;
        self.position = checkpoint.position;
        self.position_delta = checkpoint.position_delta;
        Ok(())
    }

    #[must_use]
    pub const fn logits(&self) -> &DeviceBuffer<bf16> {
        &self.logits
    }

    #[must_use]
    pub const fn position(&self) -> usize {
        self.position
    }

    #[must_use]
    pub const fn position_delta(&self) -> i32 {
        self.position_delta
    }

    fn validate(&self, tokens: &[u32], positions: &[u32], table: &BlockTable) -> Result<()> {
        let end = self
            .position
            .checked_add(tokens.len())
            .ok_or(Error::InvalidPagedKv("shared-routed prompt range overflow"))?;
        if tokens.is_empty() || positions.len() != 3 * tokens.len() || table.token_len() != end {
            return Err(Error::InvalidPagedKv(
                "shared-routed prompt differs from session position",
            ));
        }
        Ok(())
    }
}

fn allocate(backend: &CudaBackend, elements: usize) -> Result<DeviceBuffer<bf16>> {
    Ok(backend.inner.pool.allocate(&backend.inner.stream, elements)?)
}

fn bf16_tensor(tensor: &crate::CudaTensor) -> Result<&DeviceBuffer<bf16>> {
    tensor.as_bf16().ok_or_else(|| Error::DTypeMismatch {
        name: tensor.name().into(),
        expected: "BF16",
    })
}
