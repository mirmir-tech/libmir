use crate::engine::{Array, Result, Stream, gated_delta::StateArray};

/// Recurrent state of one session in one Mamba-2 layer: the causal
/// convolution history and the SSM state.
#[derive(Debug, Default)]
pub struct StateSpaceState {
    pub(super) value: Option<StateArray>,
    pub(super) history: Option<StateArray>,
    pub(super) offset: usize,
}

impl StateSpaceState {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub const fn offset(&self) -> usize {
        self.offset
    }

    pub fn reset(&mut self) {
        self.value = None;
        self.history = None;
        self.offset = 0;
    }

    pub(crate) fn snapshot(&self) -> Result<Self> {
        Ok(Self {
            value: self.value.as_ref().map(StateArray::try_clone).transpose()?,
            history: self.history.as_ref().map(StateArray::try_clone).transpose()?,
            offset: self.offset,
        })
    }

    pub(crate) fn graph_roots(&self) -> impl Iterator<Item = &Array> {
        [self.value.as_ref(), self.history.as_ref()]
            .into_iter()
            .flatten()
            .map(StateArray::graph_root)
    }

    pub(crate) fn detach_evaluated_graphs(&self, stream: &Stream) -> Result<()> {
        for array in self.graph_roots() {
            array.detach_graph(stream)?;
        }
        Ok(())
    }
}
