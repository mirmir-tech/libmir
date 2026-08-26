use super::{
    Array, Error, QuantizedArrays, Result, Stream,
    fused_gate_up::{concatenate, split_last},
};

#[derive(Debug)]
pub struct FusedAttention {
    arrays: QuantizedArrays,
    query_width: usize,
    key_width: usize,
    has_value: bool,
}

#[derive(Debug)]
pub struct AttentionProjectionOutput {
    pub query: Array,
    pub key: Array,
    pub value: Option<Array>,
}

impl FusedAttention {
    pub(crate) fn new(
        query: &QuantizedArrays,
        key: &QuantizedArrays,
        value: Option<&QuantizedArrays>,
        group_size: i32,
        bits: i32,
        stream: &Stream,
    ) -> Result<Self> {
        let query_shape = dimensions(&query.weight)?;
        let key_shape = dimensions(&key.weight)?;
        validate_pair(&query_shape, &key_shape)?;
        let mut arrays = concatenate(query, key, 0, group_size, bits, stream)?;
        if let Some(value) = value {
            let value_shape = dimensions(&value.weight)?;
            validate_pair(&query_shape, &value_shape)?;
            arrays = concatenate(&arrays, value, 0, group_size, bits, stream)?;
        }
        Ok(Self {
            arrays,
            query_width: query_shape[0],
            key_width: key_shape[0],
            has_value: value.is_some(),
        })
    }

    pub(crate) fn warm(&self, stream: &Stream) -> Result<()> {
        self.arrays.weight.async_eval(stream)?;
        self.arrays.scales.async_eval(stream)?;
        self.arrays.biases.async_eval(stream)
    }

    pub(crate) fn forward(
        &self,
        input: &Array,
        stream: &Stream,
    ) -> Result<AttentionProjectionOutput> {
        let output =
            input.quantized_matmul(&self.arrays, true, stream)?.astype_like(input, stream)?;
        let (query, remainder) = split_last(&output, self.query_width, stream)?;
        let (key, value) = split_last(&remainder, self.key_width, stream)?;
        Ok(AttentionProjectionOutput {
            query,
            key,
            value: self.has_value.then_some(value),
        })
    }
}

fn validate_pair(first: &[usize], second: &[usize]) -> Result<()> {
    if first.len() != 2 || second.len() != 2 || first[1] != second[1] {
        return Err(Error::InvalidQuantization("fused attention weights are incompatible".into()));
    }
    Ok(())
}

fn dimensions(array: &Array) -> Result<Vec<usize>> {
    Ok(array.native().shape()?.dimensions().to_vec())
}
