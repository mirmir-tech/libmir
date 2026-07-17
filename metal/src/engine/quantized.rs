use super::{Array, Result, Stream};

#[derive(Debug)]
pub struct QuantizedArrays {
    pub weight: Array,
    pub scales: Array,
    pub biases: Array,
    format: mirtal::Quantization,
}

impl QuantizedArrays {
    pub fn new(
        weight: Array,
        scales: Array,
        biases: Array,
        group_size: i32,
        bits: i32,
    ) -> Result<Self> {
        Ok(Self {
            weight,
            scales,
            biases,
            format: mirtal::Quantization::new(group_size, bits)?,
        })
    }

    pub(super) fn native(&self) -> mirtal::Quantized<'_> {
        mirtal::Quantized {
            weight: self.weight.native(),
            scales: self.scales.native(),
            biases: self.biases.native(),
            format: self.format,
        }
    }

    pub(super) const fn native_components(&self) -> [&mirtal::Array; 3] {
        [self.weight.native(), self.scales.native(), self.biases.native()]
    }
}

impl Array {
    pub fn quantize(&self, group_size: i32, bits: i32, stream: &Stream) -> Result<QuantizedArrays> {
        let arrays = stream
            .native()
            .graph()
            .quantize(self.native(), mirtal::Quantization::new(group_size, bits)?)?;
        Ok(QuantizedArrays {
            weight: Self::from_native(arrays.weight)?,
            scales: Self::from_native(arrays.scales)?,
            biases: Self::from_native(arrays.biases)?,
            format: arrays.format,
        })
    }

    pub fn quantized_matmul(
        &self,
        quantized: &QuantizedArrays,
        transpose: bool,
        stream: &Stream,
    ) -> Result<Self> {
        Self::from_native(stream.native().graph().quantized_matmul(
            self.native(),
            quantized.native(),
            transpose,
        )?)
    }

    pub fn gather_qmm(
        &self,
        quantized: &QuantizedArrays,
        rhs_indices: &Self,
        options: mirtal::GatherQmmOptions,
        stream: &Stream,
    ) -> Result<Self> {
        Self::from_native(stream.native().graph().gather_qmm(
            self.native(),
            quantized.native(),
            rhs_indices.native(),
            options,
        )?)
    }
}
