use mirtal::{Array, CompileOptions, Compiled, DType, Graph, Shape};

use super::{GatedDeltaLayer, GatedDeltaLayerConfig};
use crate::engine::{GatedDeltaState, Result, Stream};

mod batch;
mod recurrence;
mod weights;
use recurrence::Recurrence;
use weights::Weights;

#[derive(Debug)]
pub(super) struct CompiledDecode {
    graph: Compiled<3, 3>,
    #[cfg(test)]
    packed: Option<Compiled<3, 3>>,
}

impl CompiledDecode {
    pub(super) fn new(layer: &GatedDeltaLayer, stream: &Stream) -> Result<Option<Self>> {
        let Some(weights) = Weights::new(layer)? else {
            return Ok(None);
        };
        let graph = Self::compile(weights, layer.config, Recurrence::new()?, stream)?;
        #[cfg(test)]
        let packed = if layer.config.key_head_dim == 128 && layer.config.value_head_dim % 8 == 0 {
            Weights::new(layer)?
                .map(|weights| Self::compile(weights, layer.config, Recurrence::packed()?, stream))
                .transpose()?
        } else {
            None
        };
        Ok(Some(Self {
            graph,
            #[cfg(test)]
            packed,
        }))
    }

    fn compile(
        weights: Weights,
        config: GatedDeltaLayerConfig,
        kernel: Recurrence,
        stream: &Stream,
    ) -> Result<Compiled<3, 3>> {
        let graph = stream.native().compile(CompileOptions::default(), move |graph, inputs| {
            build(graph, inputs, &weights, config, &kernel)
        })?;
        Ok(graph)
    }

    fn selected(&self, stream: &Stream) -> &Compiled<3, 3> {
        #[cfg(not(test))]
        let _ = stream;
        #[cfg(test)]
        if stream.config().diagnostics.gdn_execution == crate::config::GdnExecution::PackedDecode
            && let Some(graph) = &self.packed
        {
            crate::engine::kernels::gated_delta::experiment::record();
            return graph;
        }
        &self.graph
    }

    pub(super) fn forward(
        &self,
        input: &crate::engine::Array,
        state: &mut GatedDeltaState,
        stream: &Stream,
    ) -> Result<Option<crate::engine::Array>> {
        let Some((value, convolution)) = state.compiled_decode_state() else {
            return Ok(None);
        };
        let [output, next_value, next_convolution] = self
            .selected(stream)
            .call(stream.native(), [input.native(), value.native(), convolution.native()])?;
        state.commit_compiled_decode(
            crate::engine::Array::from_native(next_value)?,
            crate::engine::Array::from_native(next_convolution)?,
        );
        Ok(Some(crate::engine::Array::from_native(output)?))
    }
}

fn build(
    graph: Graph<'_>,
    [input, state, history]: [Array; 3],
    weights: &Weights,
    config: GatedDeltaLayerConfig,
    kernel: &Recurrence,
) -> mirtal::Result<[Array; 3]> {
    let input_shape = input.shape()?;
    let input_dimensions = input_shape.dimensions();
    let batch = input_dimensions[0];
    let key_heads = usize::try_from(config.key_heads)?;
    let value_heads = usize::try_from(config.value_heads)?;
    let key_dimension = usize::try_from(config.key_head_dim)?;
    let value_dimension = usize::try_from(config.value_head_dim)?;
    let key_width = key_heads * key_dimension;
    let value_width = value_heads * value_dimension;
    let projected = weights.qkv.forward(graph, &input)?;
    let (mixed, next_history) = convolve(graph, &projected, &history, weights)?;
    let (query, key, value) = split_qkv(graph, &mixed, key_width, value_width)?;
    let query = graph.reshape(&query, &Shape::new([batch, 1, key_heads, key_dimension])?)?;
    let key = graph.reshape(&key, &Shape::new([batch, 1, key_heads, key_dimension])?)?;
    let value = graph.reshape(&value, &Shape::new([batch, 1, value_heads, value_dimension])?)?;
    let gate = graph.reshape(
        &weights.gate.forward(graph, &input)?,
        &Shape::new([batch, 1, value_heads, value_dimension])?,
    )?;
    let beta = weights.beta.forward(graph, &input)?;
    let alpha = weights.alpha.forward(graph, &input)?;
    let [recurrent, next_state] = kernel.dispatch(
        graph,
        [&query, &key, &value, &alpha, &beta, &weights.a_log, &weights.dt_bias, &state],
    )?;
    let normalized = graph.rms_norm(&recurrent, &weights.norm, config.rms_norm_eps)?;
    let normalized = graph.astype(&normalized, recurrent.dtype()?)?;
    let output = precise_gate(graph, &recurrent, &gate, &normalized)?;
    let output = graph.reshape(&output, &Shape::new([batch, 1, value_width])?)?;
    let output = weights.output.forward(graph, &output)?;
    Ok([output, next_state, next_history])
}

fn convolve(
    graph: Graph<'_>,
    input: &Array,
    history: &Array,
    weights: &Weights,
) -> mirtal::Result<(Array, Array)> {
    let combined = graph.concatenate(&[history, input], 1)?;
    let groups = i32::try_from(input.shape()?.dimensions()[2])?;
    let convolved = graph.conv1d(&combined, &weights.convolution, 1, 0, 1, groups)?;
    let kernel = weights.convolution.shape()?.dimensions()[1];
    let shape = input.shape()?;
    let dimensions = shape.dimensions();
    let history = graph.slice(&combined, &[0, 1, 0], &[dimensions[0], kernel, dimensions[2]])?;
    Ok((graph.silu(&convolved)?, history))
}

fn split_qkv(
    graph: Graph<'_>,
    input: &Array,
    key_width: usize,
    value_width: usize,
) -> mirtal::Result<(Array, Array, Array)> {
    let shape = input.shape()?;
    let dimensions = shape.dimensions();
    let slice = |start, stop| graph.slice(input, &[0, 0, start], &[dimensions[0], 1, stop]);
    Ok((
        slice(0, key_width)?,
        slice(key_width, key_width * 2)?,
        slice(key_width * 2, key_width * 2 + value_width)?,
    ))
}

fn precise_gate(
    graph: Graph<'_>,
    reference: &Array,
    gate: &Array,
    input: &Array,
) -> mirtal::Result<Array> {
    let gate = graph.astype(gate, DType::Float32)?;
    let gate = graph.multiply(&gate, &graph.sigmoid(&gate)?)?;
    let input = graph.astype(input, DType::Float32)?;
    let output = graph.multiply(&gate, &input)?;
    graph.astype(&output, reference.dtype()?)
}
