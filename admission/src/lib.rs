//! Hardware-free model architecture admission shared by libmir backends.

mod error;

pub mod cuda;
pub mod metal;

pub use error::{ArchitectureError, Result};

#[cfg(test)]
#[allow(clippy::self_named_module_files)]
mod tests;
