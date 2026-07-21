use std::collections::HashMap;

use mircuda::{DeviceBuffer, bf16};
use runtime::{backend::SamplingLogits, kv::BlockTable};
use uuid::Uuid;

use super::{
    CudaSharedRoutedLayerState, CudaSharedRoutedModelTemplate, plan::SharedRoutedExecutionPlan,
};
use crate::{
    CudaAffineOutputHead, CudaBackend, DeviceSamplerBf16, Error, Result,
    kernels::{SelectRowBf16, ShiftedRmsNorm},
};

/// One mutable CUDA session for an affine shared-routed mixed-mixer model.
pub struct CudaSharedRoutedModelSession {
    template: CudaSharedRoutedModelTemplate,
    embedding: crate::AffineQuantizedEmbedding,
    states: Vec<CudaSharedRoutedLayerState>,
    plans: HashMap<usize, SharedRoutedExecutionPlan>,
    final_norm: ShiftedRmsNorm,
    output: CudaAffineOutputHead,
    sampler: DeviceSamplerBf16,
    select_row: SelectRowBf16,
    last_hidden: DeviceBuffer<bf16>,
    normalized: DeviceBuffer<bf16>,
    logits: DeviceBuffer<bf16>,
    position: usize,
    position_delta: i32,
}

impl CudaSharedRoutedModelSession {
    pub(super) fn new(template: &CudaSharedRoutedModelTemplate) -> Result<Self> {
        let hidden = template.decoder.hidden_size;
        let vocab = template.decoder.vocab_size;
        let epsilon = template.decoder.rms_norm_eps.to_string().parse()?;
        let backend = &template.backend;
        Ok(Self {
            template: template.clone(),
            embedding: template.prepare_embedding()?,
            states: template.prepare_states()?,
            plans: HashMap::new(),
            final_norm: ShiftedRmsNorm::compile(
                &backend.inner.compiler,
                1,
                hidden,
                epsilon,
                template.norm_shift,
            )?,
            output: template.prepare_output_head()?,
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
        self.validate(tokens, positions, table)?;
        for token in tokens {
            self.embedding.validate_token(*token)?;
        }
        let count = tokens.len();
        if !self.plans.contains_key(&count) {
            self.plans.insert(count, SharedRoutedExecutionPlan::new(&self.template, count)?);
        }
        let plan = self
            .plans
            .get_mut(&count)
            .ok_or(Error::InvalidDecoderKernel("missing shared-routed execution plan"))?;
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
        Ok(&self.logits)
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

fn text_positions(start: usize, tokens: usize, delta: i32) -> Result<Vec<u32>> {
    let end = start
        .checked_add(tokens)
        .ok_or(Error::InvalidDecoderKernel("text position range overflow"))?;
    let values = (start..end)
        .map(|position| {
            let shifted = i64::try_from(position)? + i64::from(delta);
            Ok(u32::try_from(shifted)?)
        })
        .collect::<std::result::Result<Vec<_>, Error>>()?;
    Ok(values.repeat(3))
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
