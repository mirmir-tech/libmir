mod batch;
mod config;
mod load;
mod position;
mod projection;

pub use config::GatedFullAttentionConfig;
use position::rope;

use super::{
    Array, Error, KvCache, NormWeight, PagedContextMode, Result, Stream, binding::BoundLinear,
    fused::gate_up::split_last, native_paged_attention_mode,
};

#[derive(Debug)]
pub struct GatedFullAttention {
    config: GatedFullAttentionConfig,
    query: BoundLinear,
    key: BoundLinear,
    value: BoundLinear,
    #[cfg(test)]
    key_value_join: Option<super::binding::pair::KeyValueJoin>,
    output: BoundLinear,
    query_norm: NormWeight,
    key_norm: NormWeight,
}

impl GatedFullAttention {
    pub fn forward(
        &self,
        input: &Array,
        cache: &mut KvCache,
        page_min_context: usize,
        position: i32,
        causal: bool,
        stream: &Stream,
    ) -> Result<Array> {
        self.forward_with_positions(input, cache, page_min_context, position, causal, None, stream)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_with_mode(
        &self,
        input: &Array,
        cache: &mut KvCache,
        page_min_context: usize,
        position: i32,
        causal: bool,
        mode: PagedContextMode,
        stream: &Stream,
    ) -> Result<Array> {
        self.forward_inner(
            input,
            cache,
            page_min_context,
            position,
            causal,
            None,
            Some(mode),
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_positions(
        &self,
        input: &Array,
        cache: &mut KvCache,
        page_min_context: usize,
        position: i32,
        causal: bool,
        positions: Option<&Array>,
        stream: &Stream,
    ) -> Result<Array> {
        self.forward_inner(
            input, cache, page_min_context, position, causal, positions, None, stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_inner(
        &self,
        input: &Array,
        cache: &mut KvCache,
        page_min_context: usize,
        position: i32,
        causal: bool,
        positions: Option<&Array>,
        mode: Option<PagedContextMode>,
        stream: &Stream,
    ) -> Result<Array> {
        let shape = input.shape()?;
        let batch = dimension(&shape, 0, "batch")?;
        let sequence = dimension(&shape, 1, "sequence")?;
        let heads = self.config.attention_heads;
        let head_dim = self.config.head_dim;
        let query_width = heads.checked_mul(head_dim).ok_or(Error::ShapeOverflow)?;
        #[cfg(test)]
        crate::engine::probe::detail(crate::engine::probe::Stage::AttentionNorm, input)?;
        #[cfg(test)]
        crate::engine::probe::Projection::capture(
            crate::engine::probe::ProjectionKind::QueryGate,
            &self.query,
            input,
        )?;
        let projected = self.query.forward(input, stream)?.reshape(
            &[batch, sequence, heads, head_dim.checked_mul(2).ok_or(Error::ShapeOverflow)?],
            stream,
        )?;
        let (queries, gate) = split_last(&projected, usize::try_from(head_dim)?, stream)?;
        let gate = gate.reshape(&[batch, sequence, query_width], stream)?;
        let queries = self.query_norm.apply(&queries, self.config.rms_norm_eps, stream)?;

        let keys = self
            .key
            .forward(input, stream)?
            .reshape(&[batch, sequence, self.config.key_value_heads, head_dim], stream)?;
        let keys = self.key_norm.apply(&keys, self.config.rms_norm_eps, stream)?;
        let values = self
            .value
            .forward(input, stream)?
            .reshape(&[batch, sequence, self.config.key_value_heads, head_dim], stream)?
            .transpose(&[0, 2, 1, 3], stream)?;
        #[cfg(test)]
        projection::capture_inputs(&queries, &keys, &values, &gate)?;
        if !causal
            && positions.is_none()
            && let Some(output) = self.forward_scalar_persistent(
                [&queries, &gate, &keys, &values],
                cache,
                position,
                stream,
            )?
        {
            return Ok(output);
        }
        let queries = rope(&queries, &self.config, position, positions, stream)?;
        let keys = rope(&keys, &self.config, position, positions, stream)?;
        #[cfg(test)]
        projection::capture_rotated(&queries, &keys)?;
        let mode = if let Some(mode) = mode {
            mode
        } else if sequence == 1 {
            native_paged_attention_mode(
                head_dim,
                heads,
                self.config.key_value_heads,
                usize::try_from(position)? + 1,
                stream.config().cache.force_native_paged_attention,
            )
        } else {
            PagedContextMode::View
        };
        let context =
            cache.update_for_attention_mode(&keys, &values, stream, page_min_context, mode)?;
        let attended = match context.paged {
            Some(paged) => queries.paged_scaled_dot_product_attention_with_scratch(
                paged.attention(),
                paged.scratch(),
                self.config.attention_scale,
                stream,
            )?,
            None => queries.scaled_dot_product_attention(
                &context.keys,
                &context.values,
                self.config.attention_scale,
                causal,
                stream,
            )?,
        };
        let attended = attended
            .transpose(&[0, 2, 1, 3], stream)?
            .reshape(&[batch, sequence, query_width], stream)?;
        #[cfg(test)]
        crate::engine::probe::detail(crate::engine::probe::Stage::RawAttention, &attended)?;
        self.project_output(&gate.sigmoid_mul(&attended, stream)?, stream)
    }
}

fn dimension(shape: &[i32], axis: usize, name: &str) -> Result<i32> {
    shape
        .get(axis)
        .copied()
        .ok_or_else(|| Error::InvalidModel(format!("attention input has no {name} axis")))
}

#[cfg(test)]
mod tests;
