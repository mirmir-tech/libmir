use mircuda::{DeviceBuffer, bf16};

use super::{CudaAffineGatedDeltaExecution, CudaGatedDeltaState};
use crate::{
    Error, GatedDeltaInputs, Result,
    backend::{
        gated_delta::{CudaGatedDeltaBatchState, GatedDeltaBatchKey},
        scratch_pool::SharedScratch,
    },
};

impl CudaAffineGatedDeltaExecution {
    pub(crate) fn prepare_packed(
        &mut self,
        input: &DeviceBuffer<bf16>,
        states: &[&mut CudaGatedDeltaState],
        output: &DeviceBuffer<bf16>,
    ) -> Result<()> {
        let Some(first) = states.first() else {
            return Err(Error::InvalidDecoderKernel("Gated Delta packed batch is empty"));
        };
        if !self.tokens.is_multiple_of(states.len()) {
            return Err(Error::InvalidDecoderKernel("Gated Delta packed row mismatch"));
        }
        let row_tokens = self.tokens / states.len();
        self.validate(input, first, output)?;
        let supported = match &self.batch_state {
            Some(batch) => batch.lock()?.supports(states.len(), row_tokens),
            None => false,
        };
        if !supported {
            self.batch_state = Some(self.acquire_batch_state(states.len(), row_tokens)?);
        }
        let shared = self.batch()?.clone_handle();
        let mut batch = shared.lock()?;
        batch.pack(states)
    }

    /// The layer's packed states shared by its batches of every row count,
    /// allocated for `batch_capacity` rows on first use; a batch wider than
    /// the shared allocation gets states of its own.
    fn acquire_batch_state(
        &self,
        rows: usize,
        row_tokens: usize,
    ) -> Result<SharedScratch<CudaGatedDeltaBatchState>> {
        let config = self.config.state()?;
        let capacity = self.batch_capacity.max(rows);
        let key = GatedDeltaBatchKey {
            layer: self.weights.identity,
            config,
            tokens: row_tokens,
        };
        let shared = self.backend.inner.gated_delta_batches.acquire(key, || {
            CudaGatedDeltaBatchState::new(&self.backend, config, capacity, row_tokens)
        })?;
        if shared.lock()?.supports(rows, row_tokens) {
            return Ok(shared);
        }
        tracing::debug!(
            rows,
            capacity,
            "Gated Delta batch exceeds the shared packed states; allocating its own"
        );
        Ok(SharedScratch::new(CudaGatedDeltaBatchState::new(
            &self.backend, config, rows, row_tokens,
        )?))
    }

    // The scratch guard is used by the last statement; the lint cannot see it.
    #[allow(clippy::significant_drop_tightening)]
    pub(crate) fn execute_prepared_packed(
        &mut self,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let shared = self.scratch.clone_handle();
        let mut guard = shared.lock()?;
        let scratch = &mut *guard;
        let packed = self.project_qkv_gate(scratch, input)?;
        self.project_alpha_beta(scratch, input)?;
        let Self {
            backend,
            output: projection,
            transforms,
            weights,
            batch_state,
            config,
            ..
        } = self;
        let shared = batch_state
            .as_ref()
            .ok_or(Error::InvalidDecoderKernel("Gated Delta packed state was not prepared"))?
            .clone_handle();
        let mut batch_guard = shared.lock()?;
        let batch = &mut *batch_guard;
        let stream = &backend.inner.stream;
        if packed {
            batch.convolve_strided(
                scratch
                    .packed_qkv_gate
                    .as_ref()
                    .ok_or(Error::InvalidExecutionPlan("packed Gated Delta output is missing"))?,
                bf16_tensor(&weights.convolution)?,
                &mut scratch.convolved,
                config.mixed_width()? + config.value_width()?,
                0,
            )?;
        } else {
            batch.convolve(
                &scratch.mixed,
                bf16_tensor(&weights.convolution)?,
                &mut scratch.convolved,
            )?;
        }
        transforms.split_normalize(
            stream,
            &scratch.convolved,
            &mut scratch.normalized_query,
            &mut scratch.normalized_key,
            &mut scratch.value,
        )?;
        batch.recur(
            GatedDeltaInputs {
                query: &scratch.normalized_query,
                key: &scratch.normalized_key,
                value: &scratch.value,
                alpha: &scratch.alpha,
                beta: &scratch.beta,
                a_log: bf16_tensor(&weights.a_log)?,
                dt_bias: bf16_tensor(&weights.dt_bias)?,
            },
            &mut scratch.recurrent,
        )?;
        if packed {
            transforms.norm_gate_strided(
                stream,
                &scratch.recurrent,
                scratch
                    .packed_qkv_gate
                    .as_ref()
                    .ok_or(Error::InvalidExecutionPlan("packed Gated Delta output is missing"))?,
                bf16_tensor(&weights.norm)?,
                &mut scratch.gated,
                config.mixed_width()? + config.value_width()?,
                config.mixed_width()?,
            )?;
        } else {
            transforms.norm_gate(
                stream,
                &scratch.recurrent,
                &scratch.gate,
                bf16_tensor(&weights.norm)?,
                &mut scratch.gated,
            )?;
        }
        projection.execute(&scratch.gated, output)
    }

    pub(crate) fn commit_packed(&self, states: &mut [&mut CudaGatedDeltaState]) -> Result<()> {
        let shared = self.batch()?.clone_handle();
        let mut batch = shared.lock()?;
        batch.commit(states)
    }

    fn batch(&self) -> Result<&SharedScratch<CudaGatedDeltaBatchState>> {
        self.batch_state
            .as_ref()
            .ok_or(Error::InvalidDecoderKernel("Gated Delta packed state was not prepared"))
    }
}

fn bf16_tensor(tensor: &crate::CudaTensor) -> Result<&DeviceBuffer<bf16>> {
    tensor.as_bf16().ok_or_else(|| Error::DTypeMismatch {
        name: tensor.name().into(),
        expected: "BF16",
    })
}
