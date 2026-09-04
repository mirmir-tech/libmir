use super::super::{error::Result, model::LoadedModel, session::SessionState};
use crate::engine::Array;

pub(super) fn materialize(
    loaded: &LoadedModel,
    state: &SessionState,
    output: &Array,
) -> Result<()> {
    let mut roots = vec![output];
    state.cache.extend_graph_roots(&mut roots);
    loaded.stream.eval_many(&roots)?;
    loaded.settle_prefill_graph()?;
    state.cache.detach_evaluated_graphs(&loaded.stream)?;
    Ok(())
}
