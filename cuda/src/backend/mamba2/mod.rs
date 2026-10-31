//! Per-sequence Mamba-2 state: FP32 selective-scan state and BF16 causal
//! convolution history, with copy-on-request checkpoints.

use std::sync::atomic::{AtomicU64, Ordering};

use mircuda::{DeviceBuffer, bf16};

use super::{CudaBackend, recurrent::ResidentRow};
use crate::{
    Error, Result,
    kernels::{MAMBA2_CHUNK, Mamba2Geometry, Mamba2Kernels, Mamba2Launch, Mamba2Parameters},
};

mod batch;
mod checkpoint;
mod residency;
mod workspace;
pub use batch::CudaMamba2BatchState;
pub use checkpoint::CudaMamba2Checkpoint;
pub use workspace::Mamba2Workspace;

static NEXT_STATE_IDENTITY: AtomicU64 = AtomicU64::new(1);

#[derive(Debug)]
pub struct CudaMamba2State {
    backend: CudaBackend,
    kernels: Mamba2Kernels,
    state: DeviceBuffer<f32>,
    history: DeviceBuffer<bf16>,
    next_history: DeviceBuffer<bf16>,
    offset: usize,
    identity: u64,
    revision: u64,
    resident: ResidentRow,
    inline: Option<checkpoint::InlineCheckpoint>,
}

/// Algorithm advancing the selective scan over a chunk of tokens.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Mamba2ScanPath {
    /// Token-by-token recurrence; used for decode and short chunks, and the
    /// correctness oracle for the chunked path.
    Sequential,
    /// Structured state-space duality over chunks of `MAMBA2_CHUNK` tokens on
    /// tensor cores, one block per head with the state kept on chip.
    Chunked,
}

impl Mamba2ScanPath {
    /// The sequential scan when either path is sequential; keeps an explicit
    /// sequential request for every segment of a split chunk.
    const fn min_path(self, other: Self) -> Self {
        match (self, other) {
            (Self::Chunked, Self::Chunked) => Self::Chunked,
            _ => Self::Sequential,
        }
    }

    /// The chunked scan for prefill chunks when the geometry fits it,
    /// otherwise the sequential scan.
    #[must_use]
    pub fn select(tokens: usize, geometry: Mamba2Geometry) -> Self {
        if tokens >= MAMBA2_CHUNK && geometry.supports_chunked_scan() {
            Self::Chunked
        } else {
            Self::Sequential
        }
    }
}

impl CudaBackend {
    pub fn prepare_mamba2_state(&self, geometry: Mamba2Geometry) -> Result<CudaMamba2State> {
        let kernels = Mamba2Kernels::compile(&self.inner.compiler, geometry)?;
        let stream = &self.inner.stream;
        let pool = &self.inner.pool;
        let history = geometry.history_elements()?;
        Ok(CudaMamba2State {
            backend: self.clone(),
            kernels,
            state: pool.allocate_zeroed(stream, geometry.state_elements()?)?,
            history: pool.allocate_zeroed(stream, history)?,
            next_history: pool.allocate_zeroed(stream, history)?,
            offset: 0,
            identity: NEXT_STATE_IDENTITY.fetch_add(1, Ordering::Relaxed),
            revision: 0,
            resident: ResidentRow::default(),
            inline: None,
        })
    }
}

impl CudaMamba2State {
    #[must_use]
    pub const fn geometry(&self) -> Mamba2Geometry {
        self.kernels.geometry()
    }

    /// Tokens consumed since the last reset.
    #[must_use]
    pub const fn offset(&self) -> usize {
        self.offset
    }

    /// Advances the mixer by `tokens` rows of the fused input projection and
    /// writes `tokens * inner` gated, normalized values for the output
    /// projection. `workspace` must cover at least `tokens` rows.
    pub fn execute(
        &mut self,
        tokens: usize,
        projection: &DeviceBuffer<bf16>,
        parameters: Mamba2Parameters<'_>,
        workspace: &mut Mamba2Workspace,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.execute_with(
            Mamba2ScanPath::select(tokens, self.geometry()),
            tokens,
            projection,
            parameters,
            workspace,
            output,
        )
    }

    /// Like [`Self::execute`], with an explicit scan algorithm. An armed
    /// checkpoint inside the chunk splits it at the checkpoint position.
    pub fn execute_with(
        &mut self,
        path: Mamba2ScanPath,
        tokens: usize,
        projection: &DeviceBuffer<bf16>,
        parameters: Mamba2Parameters<'_>,
        workspace: &mut Mamba2Workspace,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        if workspace.tokens() < tokens {
            return Err(Error::InvalidDecoderKernel("Mamba-2 workspace is smaller than the chunk"));
        }
        let Some(before) = self.armed_checkpoint().filter(|before| (1..=tokens).contains(before))
        else {
            return self.advance_chunk(path, tokens, projection, parameters, workspace, output);
        };
        let geometry = self.geometry();
        let (width, inner) = (geometry.projection_width()?, geometry.inner()?);
        let head = Mamba2ScanPath::select(before, geometry);
        let mut head_output = output.slice(0..before * inner)?;
        self.advance_chunk(
            head,
            before,
            &projection.slice(0..before * width)?,
            parameters,
            workspace,
            &mut head_output,
        )?;
        self.stage_checkpoint()?;
        if before == tokens {
            return Ok(());
        }
        let rest = tokens - before;
        let mut rest_output = output.slice(before * inner..tokens * inner)?;
        self.advance_chunk(
            Mamba2ScanPath::select(rest, geometry).min_path(path),
            rest,
            &projection.slice(before * width..tokens * width)?,
            parameters,
            workspace,
            &mut rest_output,
        )
    }

    fn advance_chunk(
        &mut self,
        path: Mamba2ScanPath,
        tokens: usize,
        projection: &DeviceBuffer<bf16>,
        parameters: Mamba2Parameters<'_>,
        workspace: &mut Mamba2Workspace,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.materialize()?;
        let stream = &self.backend.inner.stream;
        let mut launch = Mamba2Launch {
            tokens,
            projection,
            parameters,
            state: &mut self.state,
            history: &self.history,
            next_history: &mut self.next_history,
            convolved: &mut workspace.convolved,
            scanned: &mut workspace.scanned,
            output,
        };
        match path {
            Mamba2ScanPath::Sequential => self.kernels.execute(stream, &mut launch)?,
            Mamba2ScanPath::Chunked => self.kernels.execute_chunked(stream, &mut launch)?,
        }
        std::mem::swap(&mut self.history, &mut self.next_history);
        self.advance(tokens)
    }

    pub(super) fn advance(&mut self, tokens: usize) -> Result<()> {
        self.offset = self
            .offset
            .checked_add(tokens)
            .ok_or(Error::InvalidDecoderKernel("Mamba-2 state offset overflow"))?;
        self.revision = self.revision.wrapping_add(1);
        Ok(())
    }

    /// Clears the recurrent state and convolution history.
    pub fn reset(&mut self) -> Result<()> {
        self.resident.clear();
        self.inline = None;
        let stream = &self.backend.inner.stream;
        let pool = &self.backend.inner.pool;
        self.state = pool.allocate_zeroed(stream, self.state.len())?;
        self.history = pool.allocate_zeroed(stream, self.history.len())?;
        self.next_history = pool.allocate_zeroed(stream, self.next_history.len())?;
        self.offset = 0;
        self.revision = self.revision.wrapping_add(1);
        Ok(())
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests;
