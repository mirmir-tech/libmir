use mircuda::{DeviceBuffer, Stream, bf16};

use crate::{
    CudaBackend, CudaTensor, Error, ExecutionPhase, Result,
    backend::{
        feed_forward::{FeedForward, FeedForwardExecution, LayerNormConfig},
        layer_scratch::LayerScratch,
    },
    kernels::{ElementwiseBf16, ShiftedRmsNorm},
};

/// The feed-forward of a hybrid layer together with its pre-norm weight.
#[derive(Clone, Debug)]
pub(in crate::backend) struct NormedFeedForward {
    pub(in crate::backend) norm: CudaTensor,
    pub(in crate::backend) block: FeedForward,
}

/// Residual tail of one prepared layer: `output = input + update`, or with a
/// feed-forward `h = input + update; output = h + ff(norm(h))`.
#[derive(Debug)]
pub(super) struct TailExecution {
    stream: Stream,
    norm: ShiftedRmsNorm,
    add: ElementwiseBf16,
    feed_forward: Option<(CudaTensor, FeedForwardExecution)>,
}

impl TailExecution {
    pub(super) fn new(
        backend: &CudaBackend,
        config: LayerNormConfig,
        feed_forward: Option<&NormedFeedForward>,
        tokens: usize,
        phase: ExecutionPhase,
    ) -> Result<Self> {
        let elements = tokens
            .checked_mul(config.hidden_size)
            .ok_or(Error::InvalidDecoderKernel("hybrid layer size overflow"))?;
        Ok(Self {
            stream: backend.inner.stream.clone(),
            norm: ShiftedRmsNorm::compile(
                &backend.inner.compiler,
                tokens,
                config.hidden_size,
                config.rms_norm_epsilon,
                config.norm_weight_shift,
            )?,
            add: ElementwiseBf16::compile(&backend.inner.compiler, elements)?,
            feed_forward: feed_forward
                .map(|value| {
                    Ok::<_, Error>((value.norm.clone(), value.block.prepare_phase(tokens, phase)?))
                })
                .transpose()?,
        })
    }

    /// Adds `scratch.attention`, the mixer's output projection, to `input`
    /// and applies the feed-forward.
    pub(super) fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        scratch: &mut LayerScratch,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let LayerScratch {
            attention: update,
            residual,
            normalized,
            moe,
        } = scratch;
        let Some((weight, feed_forward)) = &mut self.feed_forward else {
            return self.add.add(&self.stream, input, update, output);
        };
        let weight = weight.as_bf16().ok_or_else(|| Error::DTypeMismatch {
            name: weight.name().into(),
            expected: "BF16",
        })?;
        feed_forward.execute_residual_norm(
            &self.norm, input, update, weight, residual, normalized, moe, output, &self.add,
        )
    }
}
