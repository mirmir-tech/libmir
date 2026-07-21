use super::PagedAttentionScratch;
use crate::engine::{Array, Result, Stream};

#[derive(Debug, Clone, Copy)]
pub struct PagedAttention<'a> {
    pub key_pages: &'a Array,
    pub value_pages: &'a Array,
    pub key_scales: Option<&'a Array>,
    pub value_scales: Option<&'a Array>,
    pub page_table: &'a Array,
    pub page_dependency: &'a Array,
    pub page_size: usize,
    pub context_tokens: usize,
}

impl Array {
    pub fn scaled_dot_product_attention_with_sinks(
        &self,
        keys: &Self,
        values: &Self,
        scale: f32,
        causal: bool,
        sinks: &Self,
        stream: &Stream,
    ) -> Result<Self> {
        let mask = if causal {
            mirtal::AttentionMask::Causal
        } else {
            mirtal::AttentionMask::None
        };
        Self::from_native(stream.native().graph().scaled_dot_product_attention(
            self.native(),
            keys.native(),
            values.native(),
            mirtal::ScaledDotProductAttention { scale, mask, sinks: Some(sinks.native()) },
        )?)?
        .astype_like(self, stream)
    }

    pub fn masked_scaled_dot_product_attention_with_sinks(
        &self,
        keys: &Self,
        values: &Self,
        scale: f32,
        mask: &Self,
        sinks: &Self,
        stream: &Stream,
    ) -> Result<Self> {
        Self::from_native(stream.native().graph().scaled_dot_product_attention(
            self.native(),
            keys.native(),
            values.native(),
            mirtal::ScaledDotProductAttention {
                scale,
                mask: mirtal::AttentionMask::Array(mask.native()),
                sinks: Some(sinks.native()),
            },
        )?)?
        .astype_like(self, stream)
    }

    pub fn scaled_dot_product_attention(
        &self,
        keys: &Self,
        values: &Self,
        scale: f32,
        causal: bool,
        stream: &Stream,
    ) -> Result<Self> {
        let mask = if causal {
            mirtal::AttentionMask::Causal
        } else {
            mirtal::AttentionMask::None
        };
        Self::from_native(stream.native().graph().scaled_dot_product_attention(
            self.native(),
            keys.native(),
            values.native(),
            mirtal::ScaledDotProductAttention { scale, mask, sinks: None },
        )?)?
        .astype_like(self, stream)
    }

    pub fn masked_scaled_dot_product_attention(
        &self,
        keys: &Self,
        values: &Self,
        scale: f32,
        mask: &Self,
        stream: &Stream,
    ) -> Result<Self> {
        Self::from_native(stream.native().graph().scaled_dot_product_attention(
            self.native(),
            keys.native(),
            values.native(),
            mirtal::ScaledDotProductAttention {
                scale,
                mask: mirtal::AttentionMask::Array(mask.native()),
                sinks: None,
            },
        )?)?
        .astype_like(self, stream)
    }

    pub fn paged_scaled_dot_product_attention(
        &self,
        paged: PagedAttention<'_>,
        scale: f32,
        stream: &Stream,
    ) -> Result<Self> {
        self.paged_scaled_dot_product_attention_with_scratch(
            paged,
            &PagedAttentionScratch::default(),
            scale,
            stream,
        )
    }

    pub(crate) fn paged_scaled_dot_product_attention_with_scratch(
        &self,
        paged: PagedAttention<'_>,
        scratch: &PagedAttentionScratch,
        scale: f32,
        stream: &Stream,
    ) -> Result<Self> {
        let output = match (paged.key_scales, paged.value_scales) {
            (Some(key_scales), Some(value_scales)) => stream.quantized_paged_attention(
                [
                    self.native(),
                    paged.key_pages.native(),
                    paged.value_pages.native(),
                    key_scales.native(),
                    value_scales.native(),
                    paged.page_table.native(),
                    paged.page_dependency.native(),
                ],
                paged.page_size,
                paged.context_tokens,
                scale,
            )?,
            (None, None) => stream.paged_attention(
                [
                    self.native(),
                    paged.key_pages.native(),
                    paged.value_pages.native(),
                    paged.page_table.native(),
                    paged.page_dependency.native(),
                ],
                scratch,
                paged.page_size,
                paged.context_tokens,
                scale,
            )?,
            _ => {
                return Err(crate::engine::Error::InvalidModel(
                    "paged K/V scale arrays are incomplete".into(),
                ));
            },
        };
        Self::from_native(output)?.astype_like(self, stream)
    }
}
