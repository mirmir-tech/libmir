use mircuda::{DeviceBuffer, LaunchConfig, bf16};

use super::{QkvPostprocess, QkvPostprocessArguments};
use crate::{
    Error, Result,
    kernels::geometry::{narrow, product, require},
};

impl QkvPostprocess {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn launch<'a>(
        &self,
        inputs: [&'a DeviceBuffer<bf16>; 3],
        separate: bool,
        query_weight: &'a DeviceBuffer<bf16>,
        key_weight: &'a DeviceBuffer<bf16>,
        query_output: &'a mut DeviceBuffer<bf16>,
        key_output: &'a mut DeviceBuffer<bf16>,
        value_output: &'a mut DeviceBuffer<bf16>,
        start_position: usize,
    ) -> Result<(LaunchConfig, QkvPostprocessArguments<'a>)> {
        let query = product(self.spec.query_heads, self.spec.head_dim)?;
        let key = product(self.spec.kv_heads, self.spec.head_dim)?;
        let value = product(self.spec.kv_heads, self.spec.value_head_dim)?;
        let packed_width = query
            .checked_add(key)
            .and_then(|width| width.checked_add(value))
            .ok_or(Error::InvalidDecoderKernel("QKV packed width overflow"))?;
        if separate {
            require("Q input", product(self.spec.tokens, query)?, inputs[0].len())?;
            require("K input", product(self.spec.tokens, key)?, inputs[1].len())?;
            require("V input", product(self.spec.tokens, value)?, inputs[2].len())?;
        } else {
            require("QKV packed input", product(self.spec.tokens, packed_width)?, inputs[0].len())?;
        }
        require("Q norm weight", self.spec.head_dim, query_weight.len())?;
        require("K norm weight", self.spec.head_dim, key_weight.len())?;
        require("Q output", product(self.spec.tokens, query)?, query_output.len())?;
        require("K output", product(self.spec.tokens, key)?, key_output.len())?;
        require("V output", product(self.spec.tokens, value)?, value_output.len())?;
        Ok((
            self.config()?,
            (
                inputs[0],
                inputs[1],
                inputs[2],
                query_weight,
                key_weight,
                query_output,
                key_output,
                value_output,
                narrow(self.spec.tokens)?,
                narrow(self.spec.query_heads)?,
                narrow(self.spec.kv_heads)?,
                narrow(self.spec.head_dim)?,
                narrow(self.spec.value_head_dim)?,
                narrow(self.spec.rotary_dim)?,
                narrow(self.spec.pairing_dim)?,
                narrow(start_position)?,
                self.spec.theta,
                self.spec.epsilon,
                u32::from(separate),
                u32::from(self.spec.normalization.query),
                u32::from(self.spec.normalization.key),
                u32::from(self.spec.normalization.value),
            ),
        ))
    }
}
