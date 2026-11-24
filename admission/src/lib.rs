//! Hardware-free model architecture admission shared by libmir backends.
//!
//! Each backend lowers a semantic spec into typed operations, derives what
//! the plan requires, and admits the highest-priority runtime whose declared
//! capabilities cover it; the loaders build from that plan.

mod set;

pub mod capability;
pub mod cuda;
pub mod decoder;
mod error;
pub mod ggml;
pub mod lowering;
pub mod metal;

pub use error::{ArchitectureError, Rejection, Result, Runtime, TaskRefusal, Unlowerable};
pub use set::{Flag, Set};

#[cfg(test)]
#[allow(clippy::self_named_module_files)]
mod tests;
