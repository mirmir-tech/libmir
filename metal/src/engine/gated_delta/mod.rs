mod convolution;
mod fallback;
mod layer;
mod state;
mod update;

pub use layer::{GatedDeltaLayer, GatedDeltaLayerConfig};
use state::StateArray;

use super::{Array, Error, Result, Stream};

#[derive(Clone, Copy)]
pub struct GatedDeltaInputs<'a> {
    pub query: &'a Array,
    pub key: &'a Array,
    pub value: &'a Array,
    pub alpha: &'a Array,
    pub beta: &'a Array,
    pub a_log: &'a Array,
    pub dt_bias: &'a Array,
}

#[derive(Debug, Default)]
pub struct GatedDeltaState {
    value: Option<StateArray>,
    convolution: Option<StateArray>,
    offset: usize,
}

impl GatedDeltaState {
    pub fn new() -> Result<Self> {
        Ok(Self::default())
    }

    pub const fn offset(&self) -> Result<usize> {
        Ok(self.offset)
    }

    pub fn reset(&mut self) -> Result<()> {
        self.value = None;
        self.convolution = None;
        self.offset = 0;
        Ok(())
    }

    pub(crate) fn detach_evaluated_graphs(&self, stream: &Stream) -> Result<()> {
        for array in [&self.value, &self.convolution].into_iter().flatten() {
            array.graph_root().detach_graph(stream)?;
        }
        Ok(())
    }

    pub(crate) fn graph_roots(&self) -> impl Iterator<Item = &Array> {
        [self.value.as_ref(), self.convolution.as_ref()]
            .into_iter()
            .flatten()
            .map(StateArray::graph_root)
    }

    #[cfg(test)]
    pub(crate) fn prepare_prefix_retention(&mut self, stream: &Stream) -> Result<()> {
        if let Some(history) = &self.convolution {
            let compact = stream.native().graph().contiguous(history.native())?;
            self.convolution = Some(Array::from_native(compact)?.into());
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn convolution_root(&self) -> Option<&Array> {
        self.convolution.as_ref().map(StateArray::graph_root)
    }

    pub fn snapshot(&self) -> Result<Self> {
        Ok(Self {
            value: self.value.as_ref().map(StateArray::try_clone).transpose()?,
            convolution: self.convolution.as_ref().map(StateArray::try_clone).transpose()?,
            offset: self.offset,
        })
    }

    pub fn values(&self) -> Result<Array> {
        clone_array(self.value.as_deref())?
            .ok_or_else(|| Error::InvalidModel("Gated Delta cache has no state".into()))
    }

    pub fn convolve_silu(
        &mut self,
        input: &Array,
        weight: &Array,
        stream: &Stream,
    ) -> Result<Array> {
        convolution::convolve(self, input, weight, stream)
    }

    pub fn update(&mut self, inputs: GatedDeltaInputs<'_>, stream: &Stream) -> Result<Array> {
        update::update(self, inputs, stream)
    }

    pub(crate) fn update_normalized(
        &mut self,
        inputs: GatedDeltaInputs<'_>,
        stream: &Stream,
    ) -> Result<Array> {
        update::decode(self, inputs, true, stream)
    }

    pub(crate) fn update_fused(
        &mut self,
        inputs: GatedDeltaInputs<'_>,
        stream: &Stream,
    ) -> Result<Array> {
        update::decode(self, inputs, false, stream)
    }

    pub(crate) fn compiled_decode_state(&self) -> Option<(&Array, &Array)> {
        self.value.as_deref().zip(self.convolution.as_deref())
    }

    pub(crate) fn commit_compiled_decode(&mut self, value: Array, convolution: Array) {
        self.commit_packed_decode(value.into(), convolution.into());
    }

    #[cfg(test)]
    pub(crate) fn discard_packed_source(&mut self) -> Result<()> {
        self.value = clone_array(self.value.as_deref())?.map(StateArray::from);
        self.convolution = clone_array(self.convolution.as_deref())?.map(StateArray::from);
        Ok(())
    }

    fn compiled_batch_state(&self) -> Option<(&StateArray, &StateArray)> {
        self.value.as_ref().zip(self.convolution.as_ref())
    }

    fn commit_packed_decode(&mut self, value: StateArray, convolution: StateArray) {
        self.value = Some(value);
        self.convolution = Some(convolution);
        self.offset += 1;
    }
}

fn clone_array(input: Option<&Array>) -> Result<Option<Array>> {
    input.map(|array| Array::from_native(array.native().clone())).transpose()
}
