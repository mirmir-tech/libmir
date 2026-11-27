//! Dense (BF16/F16/F32) projections beyond a plain matrix: a bank stacked
//! from per-expert matrices, and any projection's weight as a matrix.

mod bank;
mod matrix;

pub(super) use bank::individual_bank;
