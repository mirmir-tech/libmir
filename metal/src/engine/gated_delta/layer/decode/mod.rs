use mirtal::{
    Array, CompileOptions, Compiled, DType, Dispatch, Graph, MetalKernel, OutputSpec,
    QuantizedArrays, Shape, TemplateArg,
};

use super::{GatedDeltaLayer, GatedDeltaLayerConfig};
use crate::engine::{
    GatedDeltaState, Result, Stream, kernels::gated_delta::new_gated_delta_decode_kernel,
};

mod batch;

#[derive(Debug)]
pub(super) struct CompiledDecode {
    graph: Compiled<3, 3>,
}

struct Weights {
    qkv: Linear,
    gate: Linear,
    beta: Linear,
    alpha: Linear,
    output: Linear,
    convolution: Array,
    norm: Array,
    a_log: Array,
    dt_bias: Array,
}

struct Linear {
    quantized: QuantizedArrays,
    bias: Option<Array>,
}

impl CompiledDecode {
    pub(super) fn new(layer: &GatedDeltaLayer, stream: &Stream) -> Result<Option<Self>> {
        let Some(qkv) = layer.in_proj_qkv.as_affine() else {
            return Ok(None);
        };
        let Some(gate) = layer.in_proj_z.as_affine() else {
            return Ok(None);
        };
        let Some(beta) = layer.in_proj_b.as_affine() else {
            return Ok(None);
        };
        let Some(alpha) = layer.in_proj_a.as_affine() else {
            return Ok(None);
        };
        let Some(output) = layer.out_proj.as_affine() else {
            return Ok(None);
        };
        let weights = Weights {
            qkv: Linear::new(qkv)?,
            gate: Linear::new(gate)?,
            beta: Linear::new(beta)?,
            alpha: Linear::new(alpha)?,
            output: Linear::new(output)?,
            convolution: layer.conv_weight.native().clone(),
            norm: layer.norm_weight.native_clone(),
            a_log: layer.a_log.native().clone(),
            dt_bias: layer.dt_bias.native().clone(),
        };
        let config = layer.config;
        let kernel = new_gated_delta_decode_kernel()?;
        let graph = stream.native().compile(CompileOptions::default(), move |graph, inputs| {
            build(graph, inputs, &weights, config, &kernel)
        })?;
        Ok(Some(Self { graph }))
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
            .graph
            .call(stream.native(), [input.native(), value.native(), convolution.native()])?;
        state.commit_compiled_decode(
            crate::engine::Array::from_native(next_value)?,
            crate::engine::Array::from_native(next_convolution)?,
        );
        Ok(Some(crate::engine::Array::from_native(output)?))
    }
}

impl Linear {
    fn new(linear: &crate::engine::QuantizedLinear) -> Result<Self> {
        let (quantized, bias) = linear.graph_parts()?;
        Ok(Self { quantized, bias })
    }

    fn forward(&self, graph: Graph<'_>, input: &Array) -> mirtal::Result<Array> {
        let output = graph.quantized_matmul(input, self.quantized.as_ref(), true)?;
        let output = graph.astype(&output, input.dtype()?)?;
        self.bias.as_ref().map_or(Ok(output.clone()), |bias| graph.add(&output, bias))
    }
}

fn build(
    graph: Graph<'_>,
    [input, state, history]: [Array; 3],
    weights: &Weights,
    config: GatedDeltaLayerConfig,
    kernel: &MetalKernel<8, 2>,
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
    let [recurrent, next_state] = recurrence(
        graph,
        kernel,
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

fn recurrence(
    graph: Graph<'_>,
    kernel: &MetalKernel<8, 2>,
    inputs: [&Array; 8],
) -> mirtal::Result<[Array; 2]> {
    let query = inputs[0];
    let value = inputs[2];
    let query_shape = query.shape()?;
    let value_shape = value.shape()?;
    let query_dimensions = query_shape.dimensions().to_vec();
    let value_dimensions = value_shape.dimensions().to_vec();
    kernel.dispatch_graph(
        graph,
        inputs,
        &[
            OutputSpec::new(value_shape, query.dtype()?),
            OutputSpec::new(inputs[7].shape()?, DType::Float32),
        ],
        &Dispatch::new([256, 1, value_dimensions[0] * value_dimensions[2]], [256, 1, 1]).templates(
            [
                TemplateArg::dtype("InT", query.dtype()?),
                TemplateArg::dtype("StT", DType::Float32),
                TemplateArg::int("DK", i32::try_from(query_dimensions[3])?),
                TemplateArg::int("DV", i32::try_from(value_dimensions[3])?),
                TemplateArg::int("HK", i32::try_from(query_dimensions[2])?),
                TemplateArg::int("HV", i32::try_from(value_dimensions[2])?),
                TemplateArg::bool("NORMALIZE", true),
            ],
        ),
    )
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
