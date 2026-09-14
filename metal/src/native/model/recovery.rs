use super::{LoadedModel, SessionState};
use crate::native::{
    error::{Error, Result},
    model::DecodeInput,
};

#[derive(Debug)]
pub(in crate::native) enum ExecutionRecovery {
    Ready,
    NeedsDrain(Vec<SessionState>),
}

impl ExecutionRecovery {
    pub(in crate::native) fn retain(&mut self, state: SessionState) {
        match self {
            Self::Ready => *self = Self::NeedsDrain(vec![state]),
            Self::NeedsDrain(states) => states.push(state),
        }
    }
}

impl LoadedModel {
    #[cfg(test)]
    pub(in crate::native) fn retained_execution_states(&self) -> Option<&[SessionState]> {
        match &self.recovery {
            ExecutionRecovery::Ready => None,
            ExecutionRecovery::NeedsDrain(states) => Some(states),
        }
    }

    pub(in crate::native) fn require_execution_ready(&self) -> Result<()> {
        match self.recovery {
            ExecutionRecovery::Ready => Ok(()),
            ExecutionRecovery::NeedsDrain(_) => Err(Error::ExecutionRecoveryRequired),
        }
    }

    pub(in crate::native) fn finish_decode<T>(
        &mut self,
        inputs: &[DecodeInput],
        result: Result<T>,
    ) -> Result<T> {
        self.finish_sessions(inputs.iter().map(|input| input.session), result)
    }

    pub(in crate::native) fn finish_sessions<T>(
        &mut self,
        sessions: impl IntoIterator<Item = uuid::Uuid>,
        result: Result<T>,
    ) -> Result<T> {
        match result {
            Ok(output) => Ok(output),
            Err(error) => self.fail_execution(sessions, [], error),
        }
    }

    pub(in crate::native) fn fail_execution<T>(
        &mut self,
        sessions: impl IntoIterator<Item = uuid::Uuid>,
        states: impl IntoIterator<Item = SessionState>,
        execution: Error,
    ) -> Result<T> {
        match self.retire_execution(sessions, states) {
            Ok(()) => Err(execution),
            Err(drain) => Err(Error::ExecutionRecoveryFailed {
                execution: Box::new(execution),
                drain: Box::new(drain),
            }),
        }
    }

    pub(in crate::native) fn retire_execution(
        &mut self,
        sessions: impl IntoIterator<Item = uuid::Uuid>,
        states: impl IntoIterator<Item = SessionState>,
    ) -> Result<()> {
        if matches!(self.recovery, ExecutionRecovery::Ready) {
            self.recovery = ExecutionRecovery::NeedsDrain(Vec::new());
        }
        for state in states {
            self.recovery.retain(state);
        }
        for session in sessions {
            if let Some(state) = self.sessions.remove(&session) {
                self.recovery.retain(state);
            }
        }
        self.recover_execution()
    }

    pub(in crate::native) fn recover_execution(&mut self) -> Result<()> {
        let ExecutionRecovery::NeedsDrain(retired) = &self.recovery else {
            return Ok(());
        };
        #[cfg(test)]
        if FAIL_DRAIN.replace(false) {
            return Err(Error::ExecutionRecoveryRequired);
        }
        let states = self.sessions.values().chain(retired);
        let mut roots = Vec::new();
        for state in states {
            state.cache.extend_graph_roots(&mut roots);
            if let Some(pending) = &state.pending {
                roots.push(&pending.logits);
            }
        }
        // Keep failed page ownership until every submitted or lazy alias write is
        // settled.
        self.stream.eval_many_with_paged_arenas(&roots)?;
        self.stream.synchronize()?;
        self.recovery = ExecutionRecovery::Ready;
        Ok(())
    }
}

#[cfg(test)]
thread_local! { static FAIL_DRAIN: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }

#[cfg(test)]
pub(in crate::native) fn fail_next_drain() {
    FAIL_DRAIN.set(true);
}
