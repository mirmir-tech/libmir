use super::{
    Array, Error, QuantizedArrays, Result, Stream,
    fused_gate_up::{concatenate, split_last},
};

#[derive(Debug)]
pub struct FusedExpertGateUp {
    arrays: QuantizedArrays,
    gate_width: usize,
}

impl FusedExpertGateUp {
    pub(crate) fn new(
        gate: &QuantizedArrays,
        up: &QuantizedArrays,
        group_size: i32,
        bits: i32,
        stream: &Stream,
    ) -> Result<Self> {
        let gate_shape = gate.weight.native().shape()?;
        let up_shape = up.weight.native().shape()?;
        let gate_shape = gate_shape.dimensions();
        let up_shape = up_shape.dimensions();
        if gate_shape.len() != 3
            || up_shape.len() != 3
            || gate_shape[0] != up_shape[0]
            || gate_shape[2] != up_shape[2]
        {
            return Err(Error::InvalidQuantization(
                "fused expert gate/up weights are incompatible".into(),
            ));
        }
        Ok(Self {
            arrays: concatenate(gate, up, 1, group_size, bits, stream)?,
            gate_width: gate_shape[1],
        })
    }

    pub(crate) fn warm(&self) -> Result<()> {
        self.arrays.weight.async_eval()?;
        self.arrays.scales.async_eval()?;
        self.arrays.biases.async_eval()
    }

    pub(crate) fn forward(
        &self,
        input: &Array,
        indices: &Array,
        stream: &Stream,
    ) -> Result<(Array, Array)> {
        let output = input.gather_qmm(
            &self.arrays,
            indices,
            mirtal::GatherQmmOptions { transpose: true, sorted_indices: false },
            stream,
        )?;
        let (gate, up) = split_last(&output, self.gate_width, stream)?;
        Ok((gate.astype_like(input, stream)?, up.astype_like(input, stream)?))
    }
}
