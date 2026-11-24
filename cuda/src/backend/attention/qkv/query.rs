//! The query projection of a layer that attends over an earlier layer's keys
//! and values (Gemma 4's E models): it projects no keys or values of its own.

use mircuda::{DeviceBuffer, Stream, bf16};

use super::{super::AttentionOutputProjection, QkvProjectionBuffers};
use crate::{
    CudaBackend, CudaTensor, DecodeAttentionConfig, DecodeAttentionOutputWeight, DecodeQkvWeights,
    Error, Result, RmsNormBf16,
};

#[derive(Debug)]
pub(in crate::backend::attention) struct QueryProjection {
    projection: AttentionOutputProjection,
    stream: Stream,
}

impl QueryProjection {
    pub(super) fn new(
        backend: &CudaBackend,
        config: DecodeAttentionConfig,
        tokens: usize,
        weights: Option<DecodeQkvWeights<'_>>,
    ) -> Result<Self> {
        let weight = weights.map(query_weight).transpose()?;
        Ok(Self {
            projection: AttentionOutputProjection::new_query(backend, config, tokens, weight)?,
            stream: backend.inner.stream.clone(),
        })
    }

    /// Writes the queries of the normalized `input` into the first separate
    /// buffer; the key and value buffers keep whatever they hold, which no
    /// cache store reads.
    pub(super) fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        (input_norm, norm_weight): (&RmsNormBf16, Option<&CudaTensor>),
        weights: DecodeQkvWeights<'_>,
        buffers: &mut QkvProjectionBuffers<'_>,
    ) -> Result<bool> {
        input_norm.apply(input, norm_weight, buffers.normalized)?;
        let [query, ..] = &mut *buffers.separate;
        self.projection
            .execute(&self.stream, buffers.normalized, query_weight(weights)?, query)?;
        Ok(true)
    }
}

fn query_weight(weights: DecodeQkvWeights<'_>) -> Result<DecodeAttentionOutputWeight<'_>> {
    match weights {
        DecodeQkvWeights::Query(weight) => Ok(weight),
        _ => Err(Error::InvalidExecutionPlan(
            "a layer reading shared keys and values received key and value weights",
        )),
    }
}
