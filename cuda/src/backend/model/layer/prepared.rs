use crate::{
    DecodeMoeBlockConfig, DenseSwiGluConfig, PrefillDenseSwiGlu, PrefillMoeBlockBf16,
    backend::{
        attention::AttentionLayout,
        block::PreparedDecodeMoeBlock,
        dense::{DenseLayerLayout, ExpertBankLayout, graph::PreparedDecodeDense},
    },
};

pub(in crate::backend::model) enum PreparedLayer {
    Moe(Box<PreparedDecodeMoeBlock>),
    Dense(Box<PreparedDecodeDense>),
}

impl PreparedLayer {
    pub const fn layer(&self) -> usize {
        match self {
            Self::Moe(prepared) => prepared.layer,
            Self::Dense(prepared) => prepared.block.attention.config.layer,
        }
    }
}

pub(in crate::backend::model) enum LayerPrefill<'a> {
    Moe(&'a mut PrefillMoeBlockBf16),
    Dense(&'a mut PrefillDenseSwiGlu),
}

#[expect(
    clippy::large_enum_variant,
    reason = "computed and compared per layer and prefill; a session keeps a few"
)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::backend::model) enum PrefillSignature {
    /// A layer's config and the layouts of its weights, which its plans
    /// compile in: layers of other layouts take their own plans.
    Moe(DecodeMoeBlockConfig, AttentionLayout, ExpertBankLayout),
    Dense(DenseSwiGluConfig, DenseLayerLayout),
}

pub(in crate::backend::model) struct SharedLayerPrefill {
    signature: PrefillSignature,
    plan: OwnedLayerPrefill,
}

pub(in crate::backend::model) enum OwnedLayerPrefill {
    Moe(Box<PrefillMoeBlockBf16>),
    Dense(Box<PrefillDenseSwiGlu>),
}

impl SharedLayerPrefill {
    pub const fn new(signature: &PrefillSignature, plan: OwnedLayerPrefill) -> Self {
        Self { signature: *signature, plan }
    }

    pub fn supports(&self, signature: &PrefillSignature) -> bool {
        self.signature == *signature
    }

    pub const fn borrow(&mut self) -> LayerPrefill<'_> {
        match &mut self.plan {
            OwnedLayerPrefill::Moe(plan) => LayerPrefill::Moe(plan),
            OwnedLayerPrefill::Dense(plan) => LayerPrefill::Dense(plan),
        }
    }
}

impl From<PrefillMoeBlockBf16> for OwnedLayerPrefill {
    fn from(value: PrefillMoeBlockBf16) -> Self {
        Self::Moe(Box::new(value))
    }
}

impl From<PrefillDenseSwiGlu> for OwnedLayerPrefill {
    fn from(value: PrefillDenseSwiGlu) -> Self {
        Self::Dense(Box::new(value))
    }
}
