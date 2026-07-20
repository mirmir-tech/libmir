use crate::engine::{Array, DenseLinear, ModelTensors, Result, Stream};

#[derive(Debug)]
pub(super) struct EncoderAttention {
    qkv: DenseLinear,
    output: DenseLinear,
    rope_frequencies: Option<Array>,
    heads: i32,
    head_dim: i32,
    hidden: i32,
    scale: f32,
}

impl EncoderAttention {
    pub fn load(
        tensors: &ModelTensors,
        prefix: &str,
        config: &models::layout::EncoderConfig,
        stream: &Stream,
    ) -> Result<Self> {
        let rope_frequencies = match config.rope_scaling {
            Some(models::layout::EncoderRopeScaling::Ntk { factor, mixed_b: None }) => {
                Some(Array::ntk_rope_frequencies(
                    i32::try_from(config.head_dim)?,
                    config.rope_theta.unwrap_or(10_000.0).to_string().parse()?,
                    factor.to_string().parse()?,
                    stream,
                )?)
            },
            Some(models::layout::EncoderRopeScaling::Ntk { mixed_b: Some(_), .. }) => {
                return Err(crate::engine::Error::InvalidModel(
                    "mixed NTK RoPE is not implemented for Metal sequence scoring".into(),
                ));
            },
            None => None,
        };
        let head_dim = i32::try_from(config.head_dim)?;
        Ok(Self {
            qkv: DenseLinear::load(tensors, &format!("{prefix}.qkv_proj"), stream)?,
            output: DenseLinear::load(tensors, &format!("{prefix}.o_proj"), stream)?,
            rope_frequencies,
            heads: i32::try_from(config.num_attention_heads)?,
            head_dim,
            hidden: i32::try_from(config.hidden_size)?,
            scale: head_dim.to_string().parse::<f32>()?.sqrt().recip(),
        })
    }

    pub fn forward(&self, input: &Array, stream: &Stream) -> Result<Array> {
        let sequence = input.shape()?[1];
        let qkv = self.qkv.forward(input, stream)?;
        let query = self.projection(&qkv, 0, sequence, stream)?;
        let key = self.projection(&qkv, self.hidden, sequence, stream)?;
        let value = self
            .projection(&qkv, self.hidden * 2, sequence, stream)?
            .transpose(&[0, 2, 1, 3], stream)?;
        let query = self.rotate(&query, stream)?;
        let key = self.rotate(&key, stream)?;
        let output = query.scaled_dot_product_attention(&key, &value, self.scale, false, stream)?;
        self.output.forward(
            &output
                .transpose(&[0, 2, 1, 3], stream)?
                .reshape(&[1, sequence, self.hidden], stream)?,
            stream,
        )
    }

    fn projection(&self, qkv: &Array, start: i32, sequence: i32, stream: &Stream) -> Result<Array> {
        qkv.slice(
            &[0, 0, usize::try_from(start)?],
            &[1, usize::try_from(sequence)?, usize::try_from(start + self.hidden)?],
            stream,
        )?
        .reshape(&[1, sequence, self.heads, self.head_dim], stream)
    }

    fn rotate(&self, input: &Array, stream: &Stream) -> Result<Array> {
        let input = input.transpose(&[0, 2, 1, 3], stream)?;
        if let Some(frequencies) = &self.rope_frequencies {
            input.rope_with_frequencies(self.head_dim, false, frequencies, 0, stream)
        } else {
            Ok(input)
        }
    }
}
