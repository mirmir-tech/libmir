//! Gemma 2's attention score cap, compiled into the paged attention kernels.

use mircuda::CompileOptions;

use super::{PagedAttentionSpec, compile_options};
use crate::{Error, Result};

/// A positive finite attention score cap, kept by its bits so kernel specs
/// stay hashable.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ScoreSoftcap(u32);

impl ScoreSoftcap {
    pub fn new(cap: f32) -> Result<Self> {
        if !cap.is_finite() || cap <= 0.0 {
            return Err(Error::InvalidPagedKv("attention score cap must be finite and positive"));
        }
        Ok(Self(cap.to_bits()))
    }

    #[must_use]
    pub const fn value(self) -> f32 {
        f32::from_bits(self.0)
    }
}

/// [`compile_options`] of an attention kernel, with its score cap.
pub fn attention_compile_options(spec: PagedAttentionSpec) -> Result<CompileOptions> {
    let mut options = compile_options(spec.dtype)?;
    if let Some(cap) = spec.score_softcap {
        let cap = cap.value();
        options
            .extra_options
            .push(format!("--define-macro=LIBMIR_SCORE_SOFTCAP={cap:?}f"));
    }
    Ok(options)
}
