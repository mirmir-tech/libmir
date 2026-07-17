mod arguments;
mod execution;
mod layer;
mod weights;

pub(in crate::backend) use layer::{
    CapturedDenseLayer, DenseSwiGluWeightsOwned, PreparedDecodeDense,
};
