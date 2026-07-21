use super::{Array, ModelTensors, QuantizedArrays, Result, Stream};

#[derive(Debug)]
pub struct QuantizedEmbedding {
    arrays: QuantizedArrays,
    group_size: i32,
    bits: i32,
}

impl QuantizedEmbedding {
    pub fn load(tensors: &ModelTensors, prefix: &str, group_size: i32) -> Result<Self> {
        Self::load_names(
            tensors,
            &format!("{prefix}.weight"),
            &format!("{prefix}.scales"),
            &format!("{prefix}.biases"),
            group_size,
        )
    }

    pub(in crate::engine) fn load_names(
        tensors: &ModelTensors,
        weight: &str,
        scales: &str,
        biases: &str,
        group_size: i32,
    ) -> Result<Self> {
        let weight = tensors.get(weight)?;
        let scales = tensors.get(scales)?;
        let biases = tensors.get(biases)?;
        let bits = super::linear::infer_bits(&weight, &scales, group_size)?;
        let arrays = QuantizedArrays::new(weight, scales, biases, group_size, bits)?;
        Ok(Self { arrays, group_size, bits })
    }

    pub fn lookup(&self, indices: &Array, stream: &Stream) -> Result<Array> {
        let graph = stream.native().graph();
        let quantized = self.arrays.native();
        let weight = graph.take(quantized.weight, indices.native(), 0)?;
        let scales = graph.take(quantized.scales, indices.native(), 0)?;
        let biases = graph.take(quantized.biases, indices.native(), 0)?;
        Array::from_native(graph.dequantize(mirtal::Quantized {
            weight: &weight,
            scales: &scales,
            biases: &biases,
            format: quantized.format,
        })?)
    }

    pub fn project(&self, input: &Array, stream: &Stream) -> Result<Array> {
        input.quantized_matmul(&self.arrays, true, stream)?.astype_like(input, stream)
    }

    #[must_use]
    pub fn bits(&self) -> i32 {
        self.bits
    }

    #[must_use]
    pub fn group_size(&self) -> i32 {
        self.group_size
    }
}
