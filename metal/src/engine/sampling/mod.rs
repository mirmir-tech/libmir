mod graph;

pub use graph::TopK;

use super::{Array, Result, Stream};

#[derive(Debug, Clone, Copy)]
pub struct DeviceSampling {
    pub vocab_size: usize,
    pub top_k: usize,
    pub top_p: f32,
    pub temperature: f32,
    pub draw: f32,
}

pub fn sample(logits: &Array, sampling: DeviceSampling, stream: &Stream) -> Result<Array> {
    sampling.validate()?;
    Array::from_native(graph::sample(stream.native().graph(), logits.native(), sampling)?)
}

pub fn sample_u32(logits: &Array, sampling: DeviceSampling, stream: &Stream) -> Result<u32> {
    let token = sample(logits, sampling, stream)?;
    Ok(stream.native().read_scalar_u32(token.native())?)
}

impl DeviceSampling {
    fn validate(self) -> Result<()> {
        if self.vocab_size == 0 || self.top_k == 0 || self.top_k > self.vocab_size {
            return Err(super::Error::InvalidSampling(
                "vocab size and top-k must define a non-empty range".into(),
            ));
        }
        if !self.top_p.is_finite() || self.top_p <= 0.0 || self.top_p > 1.0 {
            return Err(super::Error::InvalidSampling("top-p must be in (0, 1]".into()));
        }
        if !self.temperature.is_finite() || self.temperature <= 0.0 {
            return Err(super::Error::InvalidSampling("temperature must be positive".into()));
        }
        if !self.draw.is_finite() || self.draw < 0.0 || self.draw >= 1.0 {
            return Err(super::Error::InvalidSampling("draw must be in [0, 1)".into()));
        }
        Ok(())
    }
}

impl Array {
    pub fn top_k(&self, k: usize, vocab_size: usize, stream: &Stream) -> Result<TopK> {
        let [token_ids, scores] =
            graph::top_k(stream.native().graph(), self.native(), k, vocab_size)?;
        Ok(TopK {
            token_ids: Self::from_native(token_ids)?,
            scores: Self::from_native(scores)?,
        })
    }

    pub fn argmax_u32(&self, stream: &Stream) -> Result<u32> {
        let output = graph::argmax(stream.native().graph(), self.native())?;
        Ok(stream.native().read_scalar_u32(&output)?)
    }

    pub fn argmax(&self, stream: &Stream) -> Result<Self> {
        Self::from_native(graph::argmax(stream.native().graph(), self.native())?)
    }
}
