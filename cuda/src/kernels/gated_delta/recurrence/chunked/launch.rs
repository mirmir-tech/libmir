use mircuda::Stream;

use super::{CHUNK, GatedDeltaChunked, GatedDeltaChunkedScratch};
use crate::{
    Result,
    kernels::{
        GatedDeltaLaunch,
        geometry::{narrow, product, require},
    },
};

const NO_SCRATCH: u64 = 0;

impl GatedDeltaChunked {
    pub fn execute(
        &self,
        stream: &Stream,
        launch: &mut GatedDeltaLaunch<'_>,
        scratch: &mut GatedDeltaChunkedScratch,
    ) -> Result<()> {
        self.validate_buffers(launch)?;
        let chunks = self.spec.tokens.div_ceil(CHUNK);
        let heads = self.spec.value_heads;
        let tokens = narrow(self.spec.tokens)?;
        let gates = product(self.spec.tokens, heads)?;
        self.parameters.launch(
            stream,
            Self::config((gates.div_ceil(256), 1, 1), 8, 0)?,
            (
                launch.alpha,
                launch.beta,
                launch.a_log,
                launch.dt_bias,
                &mut *launch.decay,
                &mut *launch.update,
                tokens,
                narrow(heads)?,
            ),
        )?;
        self.cumsum.launch(
            stream,
            Self::config((chunks, heads, 1), 2, 8)?,
            (
                &*launch.decay,
                &mut scratch.cumulative_decay,
                &scratch.cu_seqlens,
                &scratch.chunk_indices,
                tokens,
                NO_SCRATCH,
                NO_SCRATCH,
            ),
        )?;
        self.kkt.launch(
            stream,
            Self::config((chunks, heads, 1), 8, 24_576)?,
            (
                launch.key,
                &*launch.update,
                &scratch.cumulative_decay,
                &mut scratch.matrix,
                &scratch.cu_seqlens,
                &scratch.chunk_indices,
                tokens,
                NO_SCRATCH,
                NO_SCRATCH,
            ),
        )?;
        self.solve.launch(
            stream,
            Self::config((chunks, heads, 1), 4, 10_240)?,
            (
                &scratch.matrix,
                &mut scratch.inverse,
                &scratch.cu_seqlens,
                &scratch.chunk_indices,
                tokens,
                NO_SCRATCH,
                NO_SCRATCH,
            ),
        )?;
        self.uw.launch(
            stream,
            Self::config((chunks, heads, 1), 2, 33_792)?,
            (
                launch.key,
                launch.value,
                &*launch.update,
                &mut scratch.w,
                &mut scratch.u,
                &scratch.inverse,
                &scratch.cumulative_decay,
                &scratch.cu_seqlens,
                &scratch.chunk_indices,
                tokens,
                NO_SCRATCH,
                NO_SCRATCH,
            ),
        )?;
        self.launch_output(stream, launch, scratch, chunks, heads, tokens)
    }

    fn launch_output(
        &self,
        stream: &Stream,
        launch: &mut GatedDeltaLaunch<'_>,
        scratch: &mut GatedDeltaChunkedScratch,
        chunks: usize,
        heads: usize,
        tokens: u32,
    ) -> Result<()> {
        let initial_state = launch.state.clone();
        self.h.launch(
            stream,
            Self::config((2, heads, 1), 4, 90_632)?,
            (
                launch.key,
                &scratch.u,
                &scratch.w,
                &mut scratch.value,
                &scratch.cumulative_decay,
                &mut scratch.chunks,
                &initial_state,
                &mut *launch.state,
                &scratch.cu_seqlens,
                &scratch.chunk_offsets,
                tokens,
                NO_SCRATCH,
                NO_SCRATCH,
            ),
        )?;
        Ok(self.o.launch(
            stream,
            Self::config((4, chunks, heads), 2, 40_960)?,
            (
                launch.query,
                launch.key,
                &scratch.value,
                &scratch.chunks,
                &scratch.cumulative_decay,
                &mut *launch.output,
                &scratch.cu_seqlens,
                &scratch.chunk_indices,
                1.0,
                tokens,
                NO_SCRATCH,
                NO_SCRATCH,
            ),
        )?)
    }

    fn validate_buffers(&self, launch: &GatedDeltaLaunch<'_>) -> Result<()> {
        let key = product(product(self.spec.tokens, self.spec.key_heads)?, self.spec.key_dim)?;
        let value =
            product(product(self.spec.tokens, self.spec.value_heads)?, self.spec.value_dim)?;
        let gates = product(self.spec.tokens, self.spec.value_heads)?;
        let state =
            product(product(self.spec.value_heads, self.spec.value_dim)?, self.spec.key_dim)?;
        require("chunked Gated Delta query", key, launch.query.len())?;
        require("chunked Gated Delta key", key, launch.key.len())?;
        require("chunked Gated Delta value", value, launch.value.len())?;
        require("chunked Gated Delta alpha", gates, launch.alpha.len())?;
        require("chunked Gated Delta beta", gates, launch.beta.len())?;
        require("chunked Gated Delta A log", self.spec.value_heads, launch.a_log.len())?;
        require("chunked Gated Delta time bias", self.spec.value_heads, launch.dt_bias.len())?;
        require("chunked Gated Delta decay", gates, launch.decay.len())?;
        require("chunked Gated Delta update", gates, launch.update.len())?;
        require("chunked Gated Delta state", state, launch.state.len())?;
        require("chunked Gated Delta output", value, launch.output.len())
    }
}
