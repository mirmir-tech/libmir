//! Layers of state-space hybrid decoders (Nemotron-H). Each logical layer is
//! a pre-normalized mixer (Mamba-2, or attention without positional
//! encoding) with a residual connection, optionally followed by a
//! normalized feed-forward with its own residual connection.

mod attention;
mod mamba;
mod tail;

pub use attention::{CudaAttentionHybridExecution, CudaAttentionHybridLayer};
pub use mamba::{CudaMamba2HybridExecution, CudaMamba2HybridLayer};
pub(in crate::backend) use mamba::{MixerScratch, MixerScratchKey};
pub(in crate::backend) use tail::NormedFeedForward;
