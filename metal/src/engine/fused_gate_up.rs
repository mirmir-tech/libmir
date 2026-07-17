use super::{Array, Error, QuantizedArrays, Result, Stream};

#[derive(Debug)]
pub struct FusedGateUp {
    arrays: QuantizedArrays,
    gate_width: usize,
}

#[derive(Debug)]
pub struct GateUpOutput {
    pub gate: Array,
    pub up: Array,
}

impl FusedGateUp {
    pub(crate) fn new(
        gate: &QuantizedArrays,
        up: &QuantizedArrays,
        group_size: i32,
        bits: i32,
        stream: &Stream,
    ) -> Result<Self> {
        let gate_shape = dimensions(&gate.weight)?;
        let up_shape = dimensions(&up.weight)?;
        if gate_shape.len() != 2 || up_shape.len() != 2 || gate_shape[1] != up_shape[1] {
            return Err(Error::InvalidQuantization(
                "fused gate/up weights are incompatible".into(),
            ));
        }
        Ok(Self {
            arrays: concatenate(gate, up, 0, group_size, bits, stream)?,
            gate_width: gate_shape[0],
        })
    }

    pub(crate) fn warm(&self) -> Result<()> {
        self.arrays.weight.async_eval()?;
        self.arrays.scales.async_eval()?;
        self.arrays.biases.async_eval()
    }

    pub(crate) fn forward(&self, input: &Array, stream: &Stream) -> Result<GateUpOutput> {
        let output = input.quantized_matmul(&self.arrays, true, stream)?;
        let (gate, up) = split_last(&output, self.gate_width, stream)?;
        Ok(GateUpOutput { gate, up })
    }

    pub(crate) fn forward_pair(&self, input: &Array, stream: &Stream) -> Result<(Array, Array)> {
        let output = self.forward(input, stream)?;
        Ok((output.gate, output.up))
    }
}

pub(super) fn concatenate(
    first: &QuantizedArrays,
    second: &QuantizedArrays,
    axis: i32,
    group_size: i32,
    bits: i32,
    stream: &Stream,
) -> Result<QuantizedArrays> {
    let graph = stream.native().graph();
    QuantizedArrays::new(
        Array::from_native(
            graph.concatenate(&[first.weight.native(), second.weight.native()], axis)?,
        )?,
        Array::from_native(
            graph.concatenate(&[first.scales.native(), second.scales.native()], axis)?,
        )?,
        Array::from_native(
            graph.concatenate(&[first.biases.native(), second.biases.native()], axis)?,
        )?,
        group_size,
        bits,
    )
}

pub(super) fn split_last(input: &Array, width: usize, stream: &Stream) -> Result<(Array, Array)> {
    let shape = input.native().shape()?;
    let rank = shape.dimensions().len();
    let total = *shape.dimensions().last().ok_or(Error::ShapeOverflow)?;
    let mut start = vec![0; rank];
    let mut stop = shape.dimensions().to_vec();
    stop[rank - 1] = width;
    let graph = stream.native().graph();
    let first = Array::from_native(graph.slice(input.native(), &start, &stop)?)?;
    start[rank - 1] = width;
    stop[rank - 1] = total;
    Ok((first, Array::from_native(graph.slice(input.native(), &start, &stop)?)?))
}

fn dimensions(array: &Array) -> Result<Vec<usize>> {
    Ok(array.native().shape()?.dimensions().to_vec())
}
