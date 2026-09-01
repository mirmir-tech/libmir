use super::{MxFp4Linear, MxFp4LinearLayout};
use crate::engine::{Dtype, Error, FusedGateUp, Result, Stream};

impl MxFp4Linear {
    pub(in crate::engine) fn fuse_gate_up(
        &self,
        up: &Self,
        stream: &Stream,
    ) -> Result<Option<FusedGateUp>> {
        if !self.fusible_with(up)? {
            return Ok(None);
        }
        FusedGateUp::new_mxfp4(
            [&self.weight, &self.scales],
            [&up.weight, &up.scales],
            self.input_features,
            self.output_features,
            up.output_features,
            stream,
        )
        .map(Some)
    }

    pub(in crate::engine) fn fused_gate_up_bytes(&self, up: &Self) -> Result<Option<usize>> {
        if !self.fusible_with(up)? {
            return Ok(None);
        }
        let mut total = 0_usize;
        for array in [&self.weight, &self.scales, &up.weight, &up.scales] {
            total = total.checked_add(array.byte_len()?).ok_or(Error::ShapeOverflow)?;
        }
        Ok(Some(total))
    }

    fn fusible_with(&self, up: &Self) -> Result<bool> {
        Ok(matches!(self.layout, MxFp4LinearLayout::Matrix)
            && matches!(up.layout, MxFp4LinearLayout::Matrix)
            && self.weight.dtype()? == Dtype::Uint32
            && up.weight.dtype()? == Dtype::Uint32
            && !self.has_bias
            && !up.has_bias
            && self.input_features == up.input_features)
    }
}
