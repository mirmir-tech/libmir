use std::time::{Duration, Instant};

use runtime::backend::{
    DecodeBatchOutput, DecodeBatchRequest, DecodeOutput, DecodeRequest, DecodeSequence,
};

use super::{build_outputs, sample_policies};
use crate::{
    Error, Result,
    engine::{
        CudaEngine,
        execution::decode_output,
        model::{ModelExecution, ModelRunner},
        profile::DecodeProfile,
    },
};

impl CudaEngine {
    pub fn decode_batch_tokens(&self, request: &DecodeBatchRequest) -> Result<DecodeBatchOutput> {
        let loaded = self.model(&request.model().id)?;
        for sequence in request.sequences() {
            loaded.require_session(sequence.session_id)?;
        }
        let waiting = Instant::now();
        let mut runner = loaded.decode_runner()?;
        let wait = waiting.elapsed();
        let started = Instant::now();
        let profile = DecodeProfile::begin(
            &self.backend,
            wait,
            request.sequences().len(),
            self.profile_decode(),
        )?;
        let mut outputs = self.decode_batch_with_runner(&mut runner, request)?;
        drop(runner);
        if let Some(profile) = profile {
            profile.finish(&self.backend, &mut outputs)?;
        }
        trace_completion(request, wait, started.elapsed());
        Ok(DecodeBatchOutput::new(outputs)?)
    }

    pub(in crate::engine) fn decode_batch_with_runner(
        &self,
        runner: &mut ModelRunner,
        request: &DecodeBatchRequest,
    ) -> Result<Vec<DecodeOutput>> {
        if let ModelExecution::Generation(generation) = &mut runner.execution
            && let Some(outputs) = generation.decode_batch(&self.backend, request.sequences())?
        {
            runner.selected = None;
            tracing::debug!(rows = outputs.len(), "executed lowered CUDA generation decode batch");
            return Ok(outputs.into_iter().map(decode_output).collect());
        }
        let mut outputs = Vec::with_capacity(request.sequences().len());
        let mut offset = 0;
        while offset < request.sequences().len() {
            let remaining = request.sequences().len() - offset;
            if let Some(rows) =
                runner.batches.as_ref().and_then(|batches| batches.largest_at_most(remaining))
            {
                outputs.extend(
                    self.execute_bucket(runner, &request.sequences()[offset..offset + rows])?,
                );
                offset += rows;
            } else {
                outputs.push(self.execute_scalar(runner, request, offset)?);
                offset += 1;
            }
        }
        Ok(outputs)
    }

    fn execute_bucket(
        &self,
        runner: &mut ModelRunner,
        sequences: &[DecodeSequence],
    ) -> Result<Vec<DecodeOutput>> {
        let rows = sequences.len();
        let tokens = sequences.iter().map(|item| item.token_id).collect::<Vec<_>>();
        let tables = sequences.iter().map(|item| &item.block_table).collect::<Vec<_>>();
        let policies = sequences.iter().map(|item| item.sampling_logits).collect::<Vec<_>>();
        let bucket = runner
            .batches
            .as_mut()
            .ok_or(Error::InvalidDecoderKernel("CUDA model has no decode batches"))?
            .get_mut(rows)?;
        bucket.decode(&tokens, &tables)?;
        let history = policies.iter().any(|policy| policy.requires_history());
        let logits = history.then(|| self.backend.read_logits(bucket.logits()?)).transpose()?;
        let sampled = sample_policies(&policies);
        let selected = if sampled.is_empty() {
            Vec::new()
        } else {
            bucket.sample(&sampled)?;
            bucket.read_sampled()?
        };
        runner.selected = None;
        let vocab = bucket.logits()?.len() / rows;
        tracing::debug!(rows, occupancy = 1.0_f64, "executed warmed CUDA decode bucket");
        build_outputs(&policies, &selected, logits.as_deref(), vocab)
    }

    fn execute_scalar(
        &self,
        runner: &mut ModelRunner,
        request: &DecodeBatchRequest,
        index: usize,
    ) -> Result<DecodeOutput> {
        let sequence = &request.sequences()[index];
        self.decode_with_runner(
            runner,
            &DecodeRequest {
                model: request.model().clone(),
                session_id: sequence.session_id,
                token_id: sequence.token_id,
                block_table: sequence.block_table.clone(),
                sampling_logits: sequence.sampling_logits,
            },
        )
    }
}

fn trace_completion(request: &DecodeBatchRequest, wait: Duration, execution: Duration) {
    tracing::debug!(
        backend = "cuda",
        rows = request.sequences().len(),
        runner_wait_ms = wait.as_secs_f64() * 1_000.0,
        execution_ms = execution.as_secs_f64() * 1_000.0,
        "completed CUDA decode batch"
    );
}
