use super::*;

impl ClampedRoutedAttention {
    pub(super) fn rotated_batch(
        &self,
        input: &Array,
        offsets: &mirtal::Array,
        stream: &Stream,
    ) -> Result<Array> {
        let input = input.transpose(&[0, 2, 1, 3], stream)?;
        Array::from_native(stream.native().graph().rope_with_frequencies_batched(
            input.native(),
            self.frequencies.native(),
            mirtal::FrequencyRopeOptions {
                dimensions: usize::try_from(self.config.head_dim)?,
                traditional: false,
                offset: offsets,
            },
        )?)?
        .astype_like(&input, stream)?
        .multiply_scalar(self.config.rope_concentration, stream)?
        .transpose(&[0, 2, 1, 3], stream)
    }
}
