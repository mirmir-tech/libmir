use mircuda::{DeviceBuffer, bf16};

use super::{CudaMamba2HybridLayer, MixerScratch, MixerWeights, bf16 as bf16_tensor};
use crate::{
    CudaBackend, CudaMamba2BatchState, CudaMamba2State, CudaTensor, DirectFp8Bf16Linear,
    DirectFp8CheckpointWeight, Error, ExecutionPhase, Result,
    backend::{
        layer_scratch::SharedLayerScratch, scratch_pool::SharedScratch,
        state_space_hybrid::tail::TailExecution,
    },
    kernels::{Mamba2Geometry, ShiftedRmsNorm},
};

/// Kernels of one Mamba-2 layer for a fixed number of rows; its row buffers
/// are shared with every Mamba-2 layer of the same shape.
#[derive(Debug)]
pub struct CudaMamba2HybridExecution {
    backend: CudaBackend,
    geometry: Mamba2Geometry,
    tokens: usize,
    input_norm: ShiftedRmsNorm,
    input_norm_weight: CudaTensor,
    input_projection: (DirectFp8Bf16Linear, DirectFp8CheckpointWeight),
    output_projection: (DirectFp8Bf16Linear, DirectFp8CheckpointWeight),
    mixer: MixerWeights,
    tail: TailExecution,
    scratch: SharedLayerScratch,
    rows: SharedScratch<MixerScratch>,
    batch: Option<CudaMamba2BatchState>,
    capacity: usize,
}

impl CudaMamba2HybridExecution {
    pub(super) fn new(
        layer: &CudaMamba2HybridLayer,
        tokens: usize,
        phase: ExecutionPhase,
    ) -> Result<Self> {
        let backend = &layer.backend;
        let geometry = layer.geometry;
        Ok(Self {
            backend: backend.clone(),
            geometry,
            tokens,
            input_norm: ShiftedRmsNorm::compile(
                &backend.inner.compiler,
                tokens,
                layer.norm.hidden_size,
                layer.norm.rms_norm_epsilon,
                layer.norm.norm_weight_shift,
            )?,
            input_norm_weight: layer.input_norm.clone(),
            input_projection: (
                layer.input_projection.prepare(backend, tokens)?,
                layer.input_projection.clone(),
            ),
            output_projection: (
                layer.output_projection.prepare(backend, tokens)?,
                layer.output_projection.clone(),
            ),
            mixer: layer.mixer.clone(),
            tail: TailExecution::new(
                backend,
                layer.norm,
                layer.feed_forward.as_ref(),
                tokens,
                phase,
            )?,
            scratch: backend.inner.layer_scratch.acquire(
                backend,
                tokens,
                layer.norm.hidden_size,
            )?,
            rows: backend.mamba2_mixer_scratch(geometry, tokens)?,
            batch: None,
            capacity: tokens,
        })
    }

    /// Advances one sequence by this execution's rows.
    pub fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        state: &mut CudaMamba2State,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.execute_ragged(input, &mut [state], &[self.tokens], output)
    }

    /// Advances several sequences whose rows are concatenated in `input`;
    /// `counts[i]` rows belong to `states[i]`, and trailing rows are padding.
    pub(crate) fn execute_ragged(
        &mut self,
        input: &DeviceBuffer<bf16>,
        states: &mut [&mut CudaMamba2State],
        counts: &[usize],
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        if states.len() != counts.len() || counts.iter().sum::<usize>() > self.tokens {
            return Err(Error::InvalidDecoderKernel("Mamba-2 ragged rows differ from the plan"));
        }
        self.project_input(input)?;
        let parameters = self.mixer.view()?;
        let (width, inner) = (self.geometry.projection_width()?, self.geometry.inner()?);
        let mut guard = self.rows.lock()?;
        let rows = &mut *guard;
        let mut offset = 0;
        for (state, &count) in states.iter_mut().zip(counts) {
            let projection = rows.projection.slice(offset * width..(offset + count) * width)?;
            let mut mixed = rows.mixed.slice(offset * inner..(offset + count) * inner)?;
            state.execute(count, &projection, parameters, &mut rows.workspace, &mut mixed)?;
            offset += count;
        }
        drop(guard);
        self.finish(input, output)
    }

    /// Rows the packed decode state is allocated for on first use.
    pub(crate) const fn set_batch_capacity(&mut self, rows: usize) {
        self.capacity = rows;
    }

    /// Host-side preparation of a packed decode step: places the sequences'
    /// states in the packed rows.
    pub(crate) fn prepare_packed(&mut self, states: &[&mut CudaMamba2State]) -> Result<()> {
        if states.len() != self.tokens {
            return Err(Error::InvalidDecoderKernel("Mamba-2 packed rows differ from the plan"));
        }
        if self.batch.is_none() {
            let capacity = self.capacity.max(self.tokens);
            self.batch = Some(self.backend.prepare_mamba2_batch_state(self.geometry, capacity)?);
        }
        self.packed()?.pack(states)
    }

    /// Device work of a packed decode step; no allocation or host sync.
    pub(crate) fn execute_prepared_packed(
        &mut self,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.project_input(input)?;
        let parameters = self.mixer.view()?;
        let batch = self
            .batch
            .as_mut()
            .ok_or(Error::InvalidDecoderKernel("Mamba-2 packed state was not packed"))?;
        let mut guard = self.rows.lock()?;
        let rows = &mut *guard;
        batch.decode(&rows.projection, parameters, &mut rows.mixed)?;
        drop(guard);
        self.finish(input, output)
    }

    pub(crate) fn commit_packed(&mut self, states: &mut [&mut CudaMamba2State]) -> Result<()> {
        self.packed()?.commit(states)
    }

    fn packed(&mut self) -> Result<&mut CudaMamba2BatchState> {
        self.batch
            .as_mut()
            .ok_or(Error::InvalidDecoderKernel("Mamba-2 packed state was not packed"))
    }

    // The guards are used by the last statement; the lint cannot see it.
    #[allow(clippy::significant_drop_tightening)]
    fn project_input(&self, input: &DeviceBuffer<bf16>) -> Result<()> {
        let mut scratch = self.scratch.lock()?;
        let mut rows = self.rows.lock()?;
        if input.len() != scratch.normalized.len() {
            return Err(Error::InvalidDecoderKernel("Mamba-2 layer input differs from the plan"));
        }
        self.input_norm.execute(
            &self.backend.inner.stream,
            input,
            bf16_tensor(&self.input_norm_weight)?,
            &mut scratch.normalized,
        )?;
        let (operation, weight) = &self.input_projection;
        operation.execute(&scratch.normalized, weight, &mut rows.projection)
    }

    // The scratch guard is used by the last statement; the lint cannot see it.
    #[allow(clippy::significant_drop_tightening)]
    fn finish(
        &mut self,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let mut guard = self.scratch.lock()?;
        let scratch = &mut *guard;
        let (operation, weight) = &self.output_projection;
        operation.execute(&self.rows.lock()?.mixed, weight, &mut scratch.attention)?;
        self.tail.execute(input, scratch, output)
    }
}
