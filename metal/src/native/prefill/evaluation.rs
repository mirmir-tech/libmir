use super::{
    super::{error::Result, model::LoadedModel, session::SessionState},
    diagnostics::{Stage, measure},
};
use crate::engine::Array;

pub(super) fn intermediate(
    loaded: &LoadedModel,
    state: &SessionState,
    output: Array,
) -> Result<()> {
    #[cfg(test)]
    if state_only_selected(loaded) {
        drop(output);
        return state_only(loaded, [state].into_iter());
    }
    let result = materialize(loaded, state, &output);
    drop(output);
    result
}

pub(super) fn intermediate_packed(
    loaded: &LoadedModel,
    states: &[&mut SessionState],
    output: Array,
) -> Result<()> {
    #[cfg(test)]
    if state_only_selected(loaded) {
        drop(output);
        return state_only(loaded, states.iter().map(|state| &**state));
    }
    let result = materialize_packed(loaded, states, &output);
    drop(output);
    result
}

#[cfg(test)]
fn state_only_selected(loaded: &LoadedModel) -> bool {
    loaded.stream.config().diagnostics.prefill_materialization
        == crate::config::PrefillMaterialization::StateOnly
}

#[cfg(test)]
fn state_only<'a>(
    loaded: &LoadedModel,
    states: impl Iterator<Item = &'a SessionState> + Clone,
) -> Result<()> {
    let mut roots = Vec::new();
    for state in states.clone() {
        state.cache.extend_prefill_graph_roots(&mut roots);
    }
    // Contiguous/sliding K/V and recurrence are explicit roots. Paged writes
    // must also finish, even without a later recurrent or output consumer.
    measure(Stage::Evaluate, || Ok(loaded.stream.eval_many_with_paged_arenas(&roots)?))?;
    loaded.settle_prefill_graph()?;
    measure(Stage::DetachState, || {
        for state in states {
            state.cache.detach_evaluated_graphs(&loaded.stream)?;
        }
        Ok(())
    })
}

pub(in crate::native) fn materialize(
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
