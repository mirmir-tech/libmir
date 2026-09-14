use super::GatedFullAttention;
use crate::engine::{Array, Result, Stream};

impl GatedFullAttention {
    pub(super) fn project_key_value(
        &self,
        input: &Array,
        sequence: i32,
        causal: bool,
        stream: &Stream,
    ) -> Result<(Array, Array)> {
        #[cfg(not(test))]
        let _ = (sequence, causal);
        #[cfg(test)]
        if !causal
            && sequence == 1
            && stream.config().diagnostics.key_value_projection
                == crate::config::KeyValueProjection::JoinedDecode
            && let Some(joined) = &self.key_value_join
        {
            return joined.forward(input, stream);
        }
        Ok((self.key.forward(input, stream)?, self.value.forward(input, stream)?))
    }

    #[cfg(test)]
    pub(super) fn capture_projection_inputs(&self, input: &Array, stream: &Stream) -> Result<()> {
        use crate::engine::probe::{self, Projection, ProjectionKind, ProjectionPair, Stage};
        probe::detail(Stage::AttentionNorm, input)?;
        Projection::capture(ProjectionKind::QueryGate, &self.query, input)?;
        ProjectionPair::capture(&self.key, &self.value, input, stream)
    }

    pub(super) fn project_output(&self, input: &Array, stream: &Stream) -> Result<Array> {
        #[cfg(test)]
        crate::engine::probe::detail(crate::engine::probe::Stage::Attended, input)?;
        #[cfg(test)]
        crate::engine::probe::Projection::capture(
            crate::engine::probe::ProjectionKind::Output,
            &self.output,
            input,
        )?;
        let output = self.output.forward(input, stream)?;
        #[cfg(test)]
        crate::engine::probe::detail(crate::engine::probe::Stage::AttentionProjection, &output)?;
        Ok(output)
    }
}

#[cfg(test)]
pub(super) fn capture_inputs(q: &Array, k: &Array, v: &Array, gate: &Array) -> Result<()> {
    use crate::engine::probe::{Stage, detail};
    for (stage, array) in
        [(Stage::Query, q), (Stage::Key, k), (Stage::Value, v), (Stage::AttentionGate, gate)]
    {
        detail(stage, array)?;
    }
    Ok(())
}

#[cfg(test)]
pub(super) fn capture_rotated(q: &Array, k: &Array) -> Result<()> {
    use crate::engine::probe::{Stage, detail};
    detail(Stage::RotatedQuery, q)?;
    detail(Stage::RotatedKey, k)
}
