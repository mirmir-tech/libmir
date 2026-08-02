use crate::{
    error::Result,
    layout::{DecoderConfig, ModelLayout},
    semantic::SemanticModelSpec,
    weights::{TensorCatalog, WeightBindingPlan},
};

#[derive(Debug, Clone, PartialEq)]
pub struct DecoderExecutionContract {
    pub semantic: SemanticModelSpec,
    pub bindings: WeightBindingPlan,
}

impl DecoderExecutionContract {
    pub fn discover(
        layout: &ModelLayout,
        decoder: &DecoderConfig,
        catalog: &TensorCatalog,
    ) -> Result<Self> {
        let semantic = SemanticModelSpec::from_layout(layout, decoder, catalog)?;
        let bindings = WeightBindingPlan::discover_from_layout(&semantic, catalog, layout)?;
        Ok(Self { semantic, bindings })
    }
}
