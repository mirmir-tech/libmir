use super::super::{Array, Result, Stream};

#[derive(Debug)]
pub(in crate::engine) struct GptqLinear {
    weight: Array,
    zero_points: Array,
    scales: Array,
    group_indices: Array,
    input: usize,
    output: usize,
    group_size: usize,
    legacy: bool,
}

impl GptqLinear {
    pub(in crate::engine) fn new(
        arrays: [Array; 4],
        input: usize,
        output: usize,
        group_size: usize,
        legacy: bool,
    ) -> Self {
        let [weight, zero_points, scales, group_indices] = arrays;
        Self {
            weight,
            zero_points,
            scales,
            group_indices,
            input,
            output,
            group_size,
            legacy,
        }
    }

    pub(in crate::engine) fn forward(&self, input: &Array, stream: &Stream) -> Result<Array> {
        stream.kernels().gptq_linear(
            stream,
            [input, &self.weight, &self.zero_points, &self.scales, &self.group_indices],
            self.input,
            self.output,
            self.group_size,
            self.legacy,
        )
    }

    pub(in crate::engine) const fn group_size(&self) -> usize {
        self.group_size
    }
}
