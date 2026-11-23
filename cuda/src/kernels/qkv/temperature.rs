//! Llama 4's query temperature as the QKV postprocess kernels compile it.

use std::num::NonZeroU32;

use mircuda::CompileOptions;
use models::semantic::QueryTemperature;

use crate::{Error, Result};

/// The query at `p` scales by `1 + beta · ln(1 + floor((p + offset) /
/// window))`; keys and values keep their scale.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QueryTemperatureScale {
    beta: f32,
    window: NonZeroU32,
    offset: u32,
}

impl QueryTemperatureScale {
    pub fn new(temperature: QueryTemperature) -> Result<Self> {
        #[allow(clippy::cast_possible_truncation)]
        let beta = temperature.beta as f32;
        let invalid = || Error::InvalidDecoderKernel("invalid query temperature");
        if !beta.is_finite() {
            return Err(invalid());
        }
        let window = u32::try_from(temperature.window)?;
        Ok(Self {
            beta,
            window: NonZeroU32::new(window).ok_or_else(invalid)?,
            offset: u32::try_from(temperature.offset)?,
        })
    }

    pub(super) fn define_macros(self, options: &mut CompileOptions) {
        options.extra_options.extend([
            format!("--define-macro=LIBMIR_QUERY_TEMPERATURE_BETA={:e}f", self.beta),
            format!("--define-macro=LIBMIR_QUERY_TEMPERATURE_WINDOW={}u", self.window),
            format!("--define-macro=LIBMIR_QUERY_TEMPERATURE_OFFSET={}u", self.offset),
        ]);
    }
}
