use models::{
    execution::{DecoderExecutionContract, TaskExecutionPlan},
    layout::{DecoderConfig, EncoderConfig, ModelLayout},
    weights::TensorCatalog,
};

use crate::native::error::Result;

pub(super) fn execution_metadata(
    task: &TaskExecutionPlan,
    layout: &ModelLayout,
    catalog: &TensorCatalog,
) -> Result<(Option<DecoderConfig>, Option<EncoderConfig>, Option<DecoderExecutionContract>)> {
    match task {
        TaskExecutionPlan::Generation { decoder }
        | TaskExecutionPlan::Embedding { decoder, .. } => {
            let contract = DecoderExecutionContract::discover(layout, decoder, catalog)?;
            Ok((Some(decoder.clone()), None, Some(contract)))
        },
        TaskExecutionPlan::SequenceScoring { encoder, .. } => {
            Ok((None, Some(encoder.clone()), None))
        },
    }
}
