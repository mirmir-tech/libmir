use super::{Array, DeviceSampling, Result, Stream};

/// Request-owned token allowlist, packed as 32 tokens per word.
#[derive(Debug, Clone, Copy)]
pub struct TokenAllowlist<'a> {
    pub words: &'a [u32],
    pub vocab: usize,
}

/// Sampling over an allowlisted distribution.
#[derive(Debug, Clone, Copy)]
pub enum MaskedSampling {
    Greedy,
    Random {
        top_k: usize,
        top_p: f32,
        temperature: f32,
        draw: f32,
    },
}

impl Array {
    /// Selects one token after rejecting every token outside `allowed`.
    /// Logit padding beyond the allowlist vocabulary is always rejected.
    pub fn masked_token_u32(
        &self,
        allowed: TokenAllowlist<'_>,
        sampling: MaskedSampling,
        stream: &Stream,
    ) -> Result<u32> {
        let vocab = self
            .shape()?
            .last()
            .copied()
            .and_then(|last| usize::try_from(last).ok())
            .ok_or_else(|| invalid("masked logits must have a vocabulary axis"))?;
        let bias = additive_mask(allowed, vocab)?;
        let bias = Self::from_f32(&bias, &[i32::try_from(vocab)?])?;
        let masked = self.add(&bias, stream)?;
        match sampling {
            MaskedSampling::Greedy => masked.argmax_u32(stream),
            MaskedSampling::Random { top_k, top_p, temperature, draw } => super::sample_u32(
                &masked,
                DeviceSampling {
                    vocab_size: vocab,
                    // Nucleus sampling without a top-k bound ranks every token.
                    top_k: if top_k == 0 && top_p < 1.0 {
                        vocab
                    } else {
                        top_k.min(vocab)
                    },
                    top_p,
                    temperature,
                    draw,
                },
                stream,
            ),
        }
    }
}

pub(super) fn additive_mask(allowed: TokenAllowlist<'_>, vocab: usize) -> Result<Vec<f32>> {
    if allowed.vocab == 0
        || allowed.vocab > vocab
        || allowed.words.len() != allowed.vocab.div_ceil(32)
    {
        return Err(invalid("token allowlist does not match the logits vocabulary"));
    }
    let bias = (0..vocab)
        .map(|token| {
            let allowed =
                token < allowed.vocab && allowed.words[token / 32] & (1_u32 << (token % 32)) != 0;
            if allowed {
                0.0
            } else {
                f32::NEG_INFINITY
            }
        })
        .collect::<Vec<_>>();
    if bias.iter().all(|value| value.is_infinite()) {
        return Err(invalid("token allowlist rejects every token"));
    }
    Ok(bias)
}

fn invalid(message: &str) -> super::super::Error {
    super::super::Error::InvalidSampling(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expands_bits_and_rejects_padding() -> Result<()> {
        let words = [0b1010_u32];
        let bias = additive_mask(TokenAllowlist { words: &words, vocab: 4 }, 6)?;

        assert_eq!(bias[..4], [f32::NEG_INFINITY, 0.0, f32::NEG_INFINITY, 0.0]);
        assert!(bias[4..].iter().all(|value| value.is_infinite()));
        Ok(())
    }

    #[test]
    fn device_selection_never_leaves_the_allowlist() -> Result<()> {
        let stream = Stream::new_gpu()?;
        // The unmasked maximum (token 0) and padding (token 5) are rejected.
        let logits = Array::from_f32(&[9.0, 1.0, 3.0, 2.0, 0.5, 8.0], &[1, 6])?;
        let words = [0b1_1110_u32];
        let allowed = TokenAllowlist { words: &words, vocab: 5 };

        assert_eq!(logits.masked_token_u32(allowed, MaskedSampling::Greedy, &stream)?, 2);
        for draw in [0.0, 0.25, 0.5, 0.75, 0.999] {
            let random = MaskedSampling::Random {
                top_k: 0,
                top_p: 0.9,
                temperature: 1.0,
                draw,
            };
            let token = logits.masked_token_u32(allowed, random, &stream)?;
            assert!((1..5).contains(&token), "draw {draw} selected {token}");
        }
        Ok(())
    }

    #[test]
    fn rejects_mismatched_or_empty_allowlists() {
        let empty = [0_u32];
        let wide = [u32::MAX; 2];
        assert!(additive_mask(TokenAllowlist { words: &empty, vocab: 4 }, 4).is_err());
        assert!(additive_mask(TokenAllowlist { words: &wide, vocab: 33 }, 32).is_err());
        assert!(additive_mask(TokenAllowlist { words: &wide, vocab: 4 }, 4).is_err());
    }
}
