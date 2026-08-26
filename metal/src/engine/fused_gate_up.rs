use super::{Array, Error, QuantizedArrays, Result, Stream};

#[derive(Debug)]
pub struct FusedGateUp {
    arrays: QuantizedArrays,
    input_width: usize,
    gate_width: usize,
    up_width: usize,
    group_size: i32,
    bits: i32,
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
        let input_width = logical_input_width(gate, group_size)?;
        if logical_input_width(up, group_size)? != input_width {
            return Err(Error::InvalidQuantization(
                "fused gate/up logical input widths are incompatible".into(),
            ));
        }
        Ok(Self {
            arrays: concatenate(gate, up, 0, group_size, bits, stream)?,
            input_width,
            gate_width: gate_shape[0],
            up_width: up_shape[0],
            group_size,
            bits,
        })
    }

    pub(crate) fn warm(&self, stream: &Stream) -> Result<()> {
        self.arrays.weight.async_eval(stream)?;
        self.arrays.scales.async_eval(stream)?;
        self.arrays.biases.async_eval(stream)
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

    pub(crate) const fn tuning_geometry(&self) -> (usize, usize, usize, i32, i32) {
        (self.input_width, self.gate_width, self.up_width, self.group_size, self.bits)
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

pub(super) fn split_interleaved_last(
    input: &Array,
    width: usize,
    stream: &Stream,
) -> Result<(Array, Array)> {
    let mut shape = input.shape()?;
    let last = shape.len().checked_sub(1).ok_or(Error::ShapeOverflow)?;
    if usize::try_from(shape[last])? != width.checked_mul(2).ok_or(Error::ShapeOverflow)? {
        return Err(Error::InvalidModel("interleaved gate/up width differs".into()));
    }
    shape[last] = i32::try_from(width)?;
    shape.push(2);
    let paired = input.reshape(&shape, stream)?;
    let mut start = vec![0; shape.len()];
    let mut stop = shape
        .iter()
        .map(|value| Ok(usize::try_from(*value)?))
        .collect::<Result<Vec<_>>>()?;
    stop[last + 1] = 1;
    let graph = stream.native().graph();
    let gate = Array::from_native(graph.slice(paired.native(), &start, &stop)?)?
        .squeeze_axis(-1, stream)?;
    start[last + 1] = 1;
    stop[last + 1] = 2;
    let up = Array::from_native(graph.slice(paired.native(), &start, &stop)?)?
        .squeeze_axis(-1, stream)?;
    Ok((gate, up))
}

fn dimensions(array: &Array) -> Result<Vec<usize>> {
    Ok(array.native().shape()?.dimensions().to_vec())
}

fn logical_input_width(arrays: &QuantizedArrays, group_size: i32) -> Result<usize> {
    let groups = dimensions(&arrays.scales)?.last().copied().ok_or(Error::ShapeOverflow)?;
    groups.checked_mul(usize::try_from(group_size)?).ok_or(Error::ShapeOverflow)
}
