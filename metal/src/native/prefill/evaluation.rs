use super::{
    super::{error::Result, model::LoadedModel, session::SessionState},
    diagnostics::{Stage, measure},
};
use crate::engine::Array;

pub(super) fn materialize(
    loaded: &LoadedModel,
    state: &SessionState,
    output: &Array,
) -> Result<()> {
    let mut roots = vec![output];
    state.cache.extend_graph_roots(&mut roots);
    measure(Stage::Evaluate, || Ok(loaded.stream.eval_many(&roots)?))?;
    loaded.settle_prefill_graph()?;
    measure(Stage::DetachState, || {
        Ok(state.cache.detach_evaluated_graphs(&loaded.stream)?)
    })?;
    Ok(())
}

pub(super) fn materialize_packed(
    loaded: &LoadedModel,
    states: &[&mut SessionState],
    output: &Array,
) -> Result<()> {
    let mut roots = vec![output];
    for state in states {
        state.cache.extend_graph_roots(&mut roots);
    }
    measure(Stage::Evaluate, || Ok(loaded.stream.eval_many(&roots)?))?;
    loaded.settle_prefill_graph()?;
    measure(Stage::DetachState, || {
        for state in states {
            state.cache.detach_evaluated_graphs(&loaded.stream)?;
        }
        Ok(())
    })
}
