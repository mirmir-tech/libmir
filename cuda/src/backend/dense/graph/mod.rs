mod arguments;
mod execution;
mod layer;
mod weights;

pub(in crate::backend) use layer::{CapturedDenseLayer, PreparedDecodeDense};
pub(in crate::backend) use weights::DenseSwiGluWeightsOwned;
