use mircuda::{DeviceBuffer, Event, Stream, bf16};

use super::{
    AffineSharedExpertMoeConfig, AffineSharedExpertMoeWeights, routed::SharedRoutedExecution,
    scratch::AffineSharedMoeScratch,
};
use crate::{
    CudaBackend, DenseRole, Error, ExecutionPhase, Result,
    backend::linear::{CheckpointProjection, CheckpointProjectionWeight, MarlinNvFp4Bf16Linear},
    kernels::{ElementwiseBf16, SigmoidMultiplyBf16},
};

#[derive(Debug)]
pub struct CudaAffineSharedExpertMoeExecution {
    backend: CudaBackend,
    config: AffineSharedExpertMoeConfig,
    tokens: usize,
    routed: SharedRoutedExecution,
    shared_gate_up: SharedGateUp,
    shared_down: CheckpointProjection,
    shared_output_gate: CheckpointProjection,
    shared_activation: ElementwiseBf16,
    output_add: ElementwiseBf16,
    sigmoid_multiply: SigmoidMultiplyBf16,
    weights: AffineSharedExpertMoeWeights,
    scratch: AffineSharedMoeScratch,
    parallel: Option<ParallelSharedExpert>,
}

#[derive(Debug)]
struct ParallelSharedExpert {
    stream: Stream,
    input_ready: Event,
    output_ready: Event,
}

#[derive(Debug)]
enum SharedGateUp {
    Separate {
        gate: Box<CheckpointProjection>,
        up: Box<CheckpointProjection>,
    },
    PackedNvFp4(MarlinNvFp4Bf16Linear),
}

impl CudaAffineSharedExpertMoeExecution {
    pub(super) fn new(
        backend: &CudaBackend,
        config: AffineSharedExpertMoeConfig,
        weights: &AffineSharedExpertMoeWeights,
        tokens: usize,
        phase: ExecutionPhase,
    ) -> Result<Self> {
        let parallel = (phase == ExecutionPhase::Decode).then(|| {
            let stream = backend.inner.auxiliary_stream.clone();
            Ok::<_, Error>(ParallelSharedExpert {
                stream,
                input_ready: backend.inner.context.create_event(false)?,
                output_ready: backend.inner.context.create_event(false)?,
            })
        });
        let parallel = parallel.transpose()?;
        let shared_backend = parallel
            .as_ref()
            .map_or_else(|| backend.clone(), |_| backend.auxiliary_backend());
        let shared = |input, output, role, weight| {
            CheckpointProjection::new(&shared_backend, tokens, input, output, role, weight)
        };
        let shared_gate_up = prepare_shared_gate_up(&shared_backend, config, weights, tokens)?;
        Ok(Self {
            backend: backend.clone(),
            config,
            tokens,
            routed: SharedRoutedExecution::new(backend, config, weights, tokens, phase)?,
            shared_gate_up,
            shared_down: shared(
                config.shared_intermediate_size,
                config.hidden_size,
                DenseRole::DenseDown,
                &weights.shared_down,
            )?,
            shared_output_gate: CheckpointProjection::new(
                &shared_backend,
                tokens,
                config.hidden_size,
                1,
                DenseRole::Router,
                &weights.shared_output_gate,
            )?,
            shared_activation: ElementwiseBf16::compile(
                &backend.inner.compiler,
                tokens * config.shared_intermediate_size,
            )?,
            output_add: ElementwiseBf16::compile(
                &backend.inner.compiler,
                tokens * config.hidden_size,
            )?,
            sigmoid_multiply: SigmoidMultiplyBf16::compile(
                &backend.inner.compiler,
                tokens,
                config.hidden_size,
            )?,
            weights: weights.clone(),
            scratch: AffineSharedMoeScratch::new(backend, config, tokens)?,
            parallel,
        })
    }

    pub fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.validate(input, output)?;
        let staged = if self.parallel.is_some() {
            self.routed.prepare_parallel(&self.backend, input)?
        } else {
            false
        };
        if let Some(parallel) = &self.parallel {
            parallel.input_ready.record(&self.backend.inner.stream)?;
            parallel.stream.wait(&parallel.input_ready)?;
        } else {
            self.routed.execute(&self.backend, &self.weights, input, &mut self.scratch)?;
        }
        self.shared(input)?;
        if let Some(parallel) = &self.parallel {
            parallel.output_ready.record(&parallel.stream)?;
            if staged {
                self.routed.execute_parallel_prepared(&self.weights, input, &mut self.scratch)?;
            } else {
                self.routed.execute(&self.backend, &self.weights, input, &mut self.scratch)?;
            }
        }
        if let Some(parallel) = &self.parallel {
            self.backend.inner.stream.wait(&parallel.output_ready)?;
        }
        self.output_add.add(
            &self.backend.inner.stream,
            &self.scratch.routed_output,
            &self.scratch.gated_shared_output,
            output,
        )
    }

    fn shared(&mut self, input: &DeviceBuffer<bf16>) -> Result<()> {
        let stream = self
            .parallel
            .as_ref()
            .map_or(&self.backend.inner.stream, |parallel| &parallel.stream);
        match &mut self.shared_gate_up {
            SharedGateUp::Separate { gate, up } => {
                gate.execute(input, &mut self.scratch.shared_gate)?;
                up.execute(input, &mut self.scratch.shared_up)?;
                self.shared_activation.gated(
                    stream,
                    &self.scratch.shared_gate,
                    &self.scratch.shared_up,
                    &mut self.scratch.shared_intermediate,
                    self.config.activation.into(),
                )?;
            },
            SharedGateUp::PackedNvFp4(operation) => {
                operation.execute(
                    input,
                    &mut self.scratch.shared_gate_up,
                    mircuda::MarlinNvFp4ThreadConfig::N128K64,
                )?;
                self.shared_activation.gated_concatenated(
                    stream,
                    &self.scratch.shared_gate_up,
                    &mut self.scratch.shared_intermediate,
                    self.config.shared_intermediate_size,
                    self.config.activation.into(),
                )?;
            },
        }
        self.shared_down
            .execute(&self.scratch.shared_intermediate, &mut self.scratch.shared_output)?;
        self.shared_output_gate.execute(input, &mut self.scratch.shared_output_gate)?;
        self.sigmoid_multiply.execute(
            stream,
            &self.scratch.shared_output,
            &self.scratch.shared_output_gate,
            &mut self.scratch.gated_shared_output,
        )
    }

    fn validate(&self, input: &DeviceBuffer<bf16>, output: &DeviceBuffer<bf16>) -> Result<()> {
        let expected = self.tokens * self.config.hidden_size;
        if input.len() != expected || output.len() != expected {
            return Err(Error::InvalidDecoderKernel("affine shared-expert MoE buffer mismatch"));
        }
        Ok(())
    }
}

fn prepare_shared_gate_up(
    backend: &CudaBackend,
    config: AffineSharedExpertMoeConfig,
    weights: &AffineSharedExpertMoeWeights,
    tokens: usize,
) -> Result<SharedGateUp> {
    if let (
        CheckpointProjectionWeight::NvFp4WeightOnly(gate),
        CheckpointProjectionWeight::NvFp4WeightOnly(up),
    ) = (&weights.shared_gate, &weights.shared_up)
        && let Some(operation) = MarlinNvFp4Bf16Linear::new_pair(backend, tokens, gate, up)?
    {
        return Ok(SharedGateUp::PackedNvFp4(operation));
    }
    Ok(SharedGateUp::Separate {
        gate: Box::new(CheckpointProjection::new(
            backend,
            tokens,
            config.hidden_size,
            config.shared_intermediate_size,
            DenseRole::DenseGateUp,
            &weights.shared_gate,
        )?),
        up: Box::new(CheckpointProjection::new(
            backend,
            tokens,
            config.hidden_size,
            config.shared_intermediate_size,
            DenseRole::DenseGateUp,
            &weights.shared_up,
        )?),
    })
}
