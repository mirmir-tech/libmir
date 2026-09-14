use crate::engine::{Array, Error, KvContext, PagedKvContext, Result, Stream};

#[derive(Clone, Copy)]
pub enum AttentionBias<'a> {
    None,
    Sinks(&'a Array),
}

enum Source<'a> {
    Paged(&'a PagedKvContext),
    View {
        keys: &'a Array,
        values: &'a Array,
        mask: Mask<'a>,
        bias: AttentionBias<'a>,
    },
}

enum Mask<'a> {
    None,
    Causal,
    Explicit(&'a Array),
}

pub struct AttentionRequest<'a> {
    query: &'a Array,
    source: Source<'a>,
    scale: f32,
}

impl<'a> AttentionRequest<'a> {
    pub(crate) fn new(
        query: &'a Array,
        context: &'a KvContext,
        scale: f32,
        causal: bool,
        bias: AttentionBias<'a>,
    ) -> Result<Self> {
        let needs_view = !matches!(bias, AttentionBias::None) || context.mask.is_some();
        if let Some(paged) = context.paged.as_ref() {
            if !needs_view {
                return Ok(Self {
                    query,
                    source: Source::Paged(paged),
                    scale,
                });
            }
            // Both-mode contexts already carry a chronological view. Native-only
            // contexts carry placeholders and cannot satisfy sinks or masks.
            let pages = paged.key_pages.native().shape()?;
            let dimensions = pages.dimensions();
            let keys = context.keys.native().shape()?;
            if dimensions.len() != 4
                || keys.dimensions() != [1, dimensions[0], paged.context_tokens, dimensions[3]]
                || context.values.native().shape()? != keys
            {
                return Err(Error::InvalidModel(
                    "attention sinks or an explicit mask require a chronological K/V view".into(),
                ));
            }
        }
        let mask = match context.mask.as_ref() {
            Some(mask) => Mask::Explicit(mask),
            None if causal => Mask::Causal,
            None => Mask::None,
        };
        let source = Source::View {
            keys: &context.keys,
            values: &context.values,
            mask,
            bias,
        };
        Ok(Self { query, source, scale })
    }

    pub(crate) fn execute(&self, stream: &Stream) -> Result<Array> {
        #[cfg(test)]
        self.capture_history()?;
        match &self.source {
            Source::Paged(paged) => self.query.paged_scaled_dot_product_attention_with_scratch(
                paged.attention(),
                paged.scratch(),
                self.scale,
                stream,
            ),
            Source::View { keys, values, mask, bias } => {
                let mask = match mask {
                    Mask::None => mirtal::AttentionMask::None,
                    Mask::Causal => mirtal::AttentionMask::Causal,
                    Mask::Explicit(mask) => mirtal::AttentionMask::Array(mask.native()),
                };
                let sinks = match bias {
                    AttentionBias::None => None,
                    AttentionBias::Sinks(sinks) => Some(sinks.native()),
                };
                Array::from_native(stream.native().graph().scaled_dot_product_attention(
                    self.query.native(),
                    keys.native(),
                    values.native(),
                    mirtal::ScaledDotProductAttention { scale: self.scale, mask, sinks },
                )?)?
                .astype_like(self.query, stream)
            },
        }
    }
}

#[cfg(test)]
mod history;
#[cfg(test)]
mod tests;
