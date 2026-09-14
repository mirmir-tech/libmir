use super::{MxFp4Linear, MxFp4LinearLayout};
use crate::engine::{Array, Dtype, Error, Result, Stream};

impl MxFp4Linear {
    pub(in crate::engine) fn forward(&self, input: &Array, stream: &Stream) -> Result<Array> {
        if !matches!(self.layout, MxFp4LinearLayout::Matrix) {
            return Err(Error::InvalidQuantization(
                "gathered MXFP4 matrix bank does not support ordinary execution".into(),
            ));
        }
        if input.dtype()? != Dtype::Bfloat16 {
            return Err(Error::InvalidQuantization("MXFP4 input must be BF16".into()));
        }
        let output = Array::from_native(stream.native().graph().mxfp4_matmul(
            input.native(),
            mirtal::MxFp4 {
                weight: self.weight.native(),
                scales: self.scales.native(),
            },
            true,
        )?)?;
        if self.has_bias {
            output.add(&self.bias, stream)
        } else {
            Ok(output)
        }
    }

    pub(in crate::engine) fn gather(
        &self,
        input: &Array,
        indices: &Array,
        sorted: bool,
        stream: &Stream,
    ) -> Result<Array> {
        let MxFp4LinearLayout::Gathered = self.layout else {
            return Err(Error::InvalidQuantization(
                "ordinary MXFP4 matrix does not support gathered execution".into(),
            ));
        };
        if input.dtype()? != Dtype::Bfloat16 || indices.dtype()? != Dtype::Uint32 {
            return Err(Error::InvalidQuantization(
                "gathered MXFP4 requires BF16 input and U32 indices".into(),
            ));
        }
        let graph = stream.native().graph();
        let output = graph.gather_mxfp4(
            input.native(),
            mirtal::MxFp4 {
                weight: self.weight.native(),
                scales: self.scales.native(),
            },
            indices.native(),
            mirtal::GatherQmmOptions { transpose: true, sorted_indices: sorted },
        )?;
        let output = if self.has_bias {
            let bias = graph.take(self.bias.native(), indices.native(), 0)?;
            graph.add(&output, &graph.expand_dims(&bias, &[-2])?)?
        } else {
            output
        };
        Array::from_native(output)
    }

    pub(in crate::engine) const fn has_bias(&self) -> bool {
        self.has_bias
    }
}
