use super::PooledLayer;
use crate::{CudaTensor, Error, Result, backend::vision::pooled::scratch::PooledScratch};

impl PooledLayer {
    pub(in crate::backend::vision::pooled) fn execute(
        &mut self,
        scratch: &mut PooledScratch,
        positions: &mircuda::DeviceBuffer<u32>,
    ) -> Result<()> {
        let stream = &self.backend.inner.stream;
        self.input_norm.execute(
            stream,
            &scratch.hidden_a,
            weight(&self.input_weight)?,
            &mut scratch.normalized,
        )?;
        self.query.execute(&scratch.normalized, &mut scratch.query)?;
        self.key.execute(&scratch.normalized, &mut scratch.key)?;
        self.value.execute(&scratch.normalized, &mut scratch.value)?;
        self.query_norm.execute(
            stream,
            &scratch.query,
            weight(&self.query_weight)?,
            &mut scratch.query_rope,
        )?;
        self.key_norm.execute(
            stream,
            &scratch.key,
            weight(&self.key_weight)?,
            &mut scratch.key_rope,
        )?;
        self.value_norm.execute(stream, &scratch.value, &mut scratch.key)?;
        self.rope_query
            .execute(stream, &scratch.query_rope, positions, &mut scratch.query)?;
        self.rope_key
            .execute(stream, &scratch.key_rope, positions, &mut scratch.value)?;
        self.attention.execute(
            stream,
            &scratch.query,
            &scratch.value,
            &scratch.key,
            &mut scratch.hidden_b,
        )?;
        self.output.execute(&scratch.hidden_b, &mut scratch.normalized)?;
        self.post_attention_norm.execute(
            stream,
            &scratch.normalized,
            weight(&self.post_attention_weight)?,
            &mut scratch.hidden_b,
        )?;
        self.elementwise_hidden.add(
            stream,
            &scratch.hidden_a,
            &scratch.hidden_b,
            &mut scratch.normalized,
        )?;
        self.pre_feedforward_norm.execute(
            stream,
            &scratch.normalized,
            weight(&self.pre_feedforward_weight)?,
            &mut scratch.hidden_a,
        )?;
        self.gate.execute(&scratch.hidden_a, &mut scratch.intermediate_a)?;
        self.up.execute(&scratch.hidden_a, &mut scratch.intermediate_b)?;
        self.elementwise_intermediate.gelu(
            stream,
            &scratch.intermediate_a,
            &mut scratch.intermediate_c,
            true,
        )?;
        self.elementwise_intermediate.multiply(
            stream,
            &scratch.intermediate_c,
            &scratch.intermediate_b,
            &mut scratch.intermediate_a,
        )?;
        self.down.execute(&scratch.intermediate_a, &mut scratch.hidden_a)?;
        self.post_feedforward_norm.execute(
            stream,
            &scratch.hidden_a,
            weight(&self.post_feedforward_weight)?,
            &mut scratch.hidden_b,
        )?;
        self.elementwise_hidden.add(
            stream,
            &scratch.normalized,
            &scratch.hidden_b,
            &mut scratch.hidden_a,
        )
    }
}

fn weight(tensor: &CudaTensor) -> Result<&mircuda::DeviceBuffer<mircuda::bf16>> {
    tensor.as_bf16().ok_or_else(|| Error::DTypeMismatch {
        name: tensor.name().into(),
        expected: "BF16",
    })
}
