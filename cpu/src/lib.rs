//! CPU inference backend for libmir, built on `mircup`.
//!
//! It currently runs Laya decision checkpoints; generation stays on the
//! accelerator backends.

mod decision;
mod error;

pub use decision::CpuDecisionModel;
pub use error::{Error, Result};
