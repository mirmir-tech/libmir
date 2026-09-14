use super::{Array, Result, SharedExpertMoe, Stream};

impl SharedExpertMoe {
    pub(super) fn aligned_routed(
        &self,
        input: &Array,
        indices: &Array,
        weights: &Array,
        stream: &Stream,
    ) -> Result<Option<Array>> {
        // Match the disabled-tuning GroupedFused fallback exactly. Decode and
        // short-tail execution retain their original route and reduction.
        if stream.config().diagnostics.moe_prefill != crate::config::MoePrefill::Aligned
            || indices.native().len() <= 1024
        {
            return Ok(None);
        }
        let grouped = input.align_expert_inputs(
            indices,
            self.config.expert_count,
            stream.kernels().aligned_group(),
            stream,
        )?;
        let output = self.routed_mlp(&grouped.input, &grouped.indices, true, false, stream)?;
        Ok(Some(grouped.restore_weighted(&output, weights, stream)?))
    }
}
