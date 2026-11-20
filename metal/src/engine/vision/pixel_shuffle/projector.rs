//! The encoder's optional final layer norm, the pixel shuffle folding each
//! `factor`×`factor` block of patches into one token (channel blocks over the
//! block's rows, then its columns), and the RMS-normalized squared-rectifier
//! projector into the decoder's hidden size.

use models::layout::PixelShuffleVisionConfig;

use super::dimension;
use crate::engine::{Array, DenseLinear, LayerNorm, ModelTensors, NormWeight, Result, Stream};

#[derive(Debug)]
pub(super) struct Projector {
    final_norm: Option<LayerNorm>,
    norm: NormWeight,
    first: DenseLinear,
    output: DenseLinear,
    norm_eps: f32,
    prefix_tokens: usize,
    hidden: usize,
    factor: usize,
}

impl Projector {
    pub(super) fn load(
        tensors: &ModelTensors,
        config: &PixelShuffleVisionConfig,
        stream: &Stream,
    ) -> Result<Self> {
        let final_norm = config
            .final_layer_norm
            .then(|| {
                LayerNorm::load(
                    tensors,
                    "vision_projector.vision_final_layernorm",
                    config.layer_norm_eps.to_string().parse()?,
                )
            })
            .transpose()?;
        Ok(Self {
            final_norm,
            norm: NormWeight::load_name(tensors, "mlp1.0.weight")?,
            first: DenseLinear::load(tensors, "mlp1.1", stream)?,
            output: DenseLinear::load(tensors, "mlp1.3", stream)?,
            norm_eps: config.projector_norm_eps.to_string().parse()?,
            prefix_tokens: config.prefix_tokens,
            hidden: config.hidden_size,
            factor: config.shuffle_factor,
        })
    }

    pub(super) fn forward(
        &self,
        encoded: &Array,
        rows: usize,
        cols: usize,
        stream: &Stream,
    ) -> Result<Array> {
        let normalized_encoder;
        let encoded = match &self.final_norm {
            Some(norm) => {
                normalized_encoder = norm.forward(encoded, stream)?;
                &normalized_encoder
            },
            None => encoded,
        };
        let patches = rows * cols;
        let f = dimension(self.factor, "shuffle factor")?;
        let hidden = dimension(self.hidden, "hidden size")?;
        let (folded_rows, folded_cols) =
            (dimension(rows / self.factor, "rows")?, dimension(cols / self.factor, "cols")?);
        let folded = encoded
            .slice(
                &[0, self.prefix_tokens, 0],
                &[1, self.prefix_tokens + patches, self.hidden],
                stream,
            )?
            .reshape(&[folded_rows, f, folded_cols, f, hidden], stream)?
            .transpose(&[0, 2, 1, 3, 4], stream)?
            .reshape(&[1, folded_rows * folded_cols, f * f * hidden], stream)?;
        let normalized = self.norm.apply(&folded, self.norm_eps, stream)?;
        self.output
            .forward(&self.first.forward(&normalized, stream)?.relu_squared(stream)?, stream)
    }
}
