mod backend;
#[cfg(test)]
mod benchmark;
mod error;
mod model;
mod output;
mod prefill;
mod prefix;
mod session;
mod step;
mod trace;

pub use backend::{MetalBackend, MetalMemoryStats};
