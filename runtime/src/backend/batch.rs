use uuid::Uuid;

use super::{DecodeOutput, DecodeRequest, ModelHandle, SamplingLogits};
use crate::{Result, RuntimeError, kv::BlockTable};

/// One sequence participating in a homogeneous decode batch.
#[derive(Debug, Clone)]
pub struct DecodeSequence {
    pub session_id: Uuid,
    pub token_id: u32,
    pub block_table: BlockTable,
    pub sampling_logits: SamplingLogits,
}

/// A non-empty decode batch targeting exactly one loaded model.
#[derive(Debug, Clone)]
pub struct DecodeBatchRequest {
    model: ModelHandle,
    sequences: Vec<DecodeSequence>,
}

impl DecodeBatchRequest {
    /// Creates a homogeneous batch. An empty sequence list is rejected.
    pub fn new(model: ModelHandle, sequences: Vec<DecodeSequence>) -> Result<Self> {
        if sequences.is_empty() {
            return Err(RuntimeError::Scheduler("decode batch cannot be empty".into()));
        }
        Ok(Self { model, sequences })
    }

    #[must_use]
    pub const fn model(&self) -> &ModelHandle {
        &self.model
    }

    #[must_use]
    pub fn sequences(&self) -> &[DecodeSequence] {
        &self.sequences
    }

    #[must_use]
    pub fn into_parts(self) -> (ModelHandle, Vec<DecodeSequence>) {
        (self.model, self.sequences)
    }
}

impl TryFrom<Vec<DecodeRequest>> for DecodeBatchRequest {
    type Error = RuntimeError;

    fn try_from(requests: Vec<DecodeRequest>) -> Result<Self> {
        let model = requests
            .first()
            .map(|request| request.model.clone())
            .ok_or_else(|| RuntimeError::Scheduler("decode batch cannot be empty".into()))?;
        if requests
            .iter()
            .any(|request| request.model.id != model.id || request.model.backend != model.backend)
        {
            return Err(RuntimeError::Scheduler(
                "decode batch must target one loaded model".into(),
            ));
        }
        let sequences = requests
            .into_iter()
            .map(|request| DecodeSequence {
                session_id: request.session_id,
                token_id: request.token_id,
                block_table: request.block_table,
                sampling_logits: request.sampling_logits,
            })
            .collect();
        Self::new(model, sequences)
    }
}

/// Outputs retain the exact input sequence order.
#[derive(Debug, Clone)]
pub struct DecodeBatchOutput {
    outputs: Vec<DecodeOutput>,
}

impl DecodeBatchOutput {
    pub fn new(outputs: Vec<DecodeOutput>) -> Result<Self> {
        if outputs.is_empty() {
            return Err(RuntimeError::Scheduler("decode batch output cannot be empty".into()));
        }
        Ok(Self { outputs })
    }

    #[must_use]
    pub fn outputs(&self) -> &[DecodeOutput] {
        &self.outputs
    }

    #[must_use]
    pub fn into_outputs(self) -> Vec<DecodeOutput> {
        self.outputs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(model: &str) -> DecodeRequest {
        DecodeRequest {
            model: ModelHandle { id: model.into(), backend: "test".into() },
            session_id: Uuid::new_v4(),
            token_id: 1,
            block_table: BlockTable::default(),
            sampling_logits: SamplingLogits::None,
        }
    }

    #[test]
    fn rejects_an_empty_batch() {
        assert!(DecodeBatchRequest::try_from(Vec::new()).is_err());
    }

    #[test]
    fn rejects_requests_for_different_models() {
        assert!(DecodeBatchRequest::try_from(vec![request("first"), request("second")]).is_err());
    }

    #[test]
    fn preserves_sequence_order() -> Result<()> {
        let first = request("model");
        let second = request("model");
        let expected = [first.session_id, second.session_id];
        let batch = DecodeBatchRequest::try_from(vec![first, second])?;
        let actual: Vec<_> = batch.sequences().iter().map(|item| item.session_id).collect();
        assert_eq!(actual, expected);
        Ok(())
    }
}
