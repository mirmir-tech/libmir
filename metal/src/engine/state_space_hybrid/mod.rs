//! Decoders interleaving Mamba-2 selective state-space mixers with plain
//! attention and sigmoid-routed `relu²` experts (Nemotron-H).

mod attention;
mod batch;
mod feed_forward;
mod layer;
mod mamba;
mod model;
#[cfg(test)]
mod tests;

pub use mamba::StateSpaceState;
pub use model::StateSpaceHybridModel;
