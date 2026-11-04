use super::{Array, DecoderModel, Error, Result, Stream};

impl DecoderModel {
    pub(crate) fn prefill_logits(&self, hidden: &Array, stream: &Stream) -> Result<Array> {
        let shape = hidden.shape()?;
        let [batch, sequence, width] = shape.as_slice() else {
            return Err(Error::InvalidModel("prefill output must have three axes".into()));
        };
        if *sequence < 1 {
            return Err(Error::InvalidModel("prefill output is empty".into()));
        }
        let last = hidden.slice(
            &[0, usize::try_from(sequence - 1)?, 0],
            &[usize::try_from(*batch)?, usize::try_from(*sequence)?, usize::try_from(*width)?],
            stream,
        )?;
        self.execution.prefill_logits(&last, stream)
    }
}
