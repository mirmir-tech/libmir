use super::{Array, FusedGateUp, QuantizedArrays, Result, Stream};

#[derive(Debug)]
pub struct FusedKeyValue {
    pair: FusedGateUp,
}

impl FusedKeyValue {
    pub(crate) fn new(
        key: &QuantizedArrays,
        value: &QuantizedArrays,
        group_size: i32,
        bits: i32,
        stream: &Stream,
    ) -> Result<Self> {
        Ok(Self {
            pair: FusedGateUp::new(key, value, group_size, bits, stream)?,
        })
    }

    pub(crate) fn warm(&self, stream: &Stream) -> Result<()> {
        self.pair.warm(stream)
    }

    pub(crate) fn forward(&self, input: &Array, stream: &Stream) -> Result<(Array, Array)> {
        self.pair.forward_pair(input, stream)
    }
}
