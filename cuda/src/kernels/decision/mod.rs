//! Kernels of the Laya decision model.

mod attention;
mod element;
mod elementwise;
mod layout;

pub use attention::{DecisionAttention, DecisionAttentionInput, DecisionWindow};
pub use element::DecisionElement;
pub use elementwise::DecisionElementwise;
pub use layout::{DecisionLayout, HeadLayout, RopeTables, ScoreRows};
use mircuda::LaunchConfig;

use crate::{Error, Result};

const BLOCK: u32 = 256;

fn rows_launch(rows: usize) -> Result<LaunchConfig> {
    Ok(LaunchConfig {
        grid: (u32::try_from(rows)?, 1, 1),
        block: (BLOCK, 1, 1),
        shared_memory_bytes: 0,
    })
}

fn elements_launch(elements: usize) -> Result<LaunchConfig> {
    Ok(LaunchConfig::for_elements(elements, BLOCK)?)
}

fn require<T: mircuda::DeviceElement>(
    buffer: &mircuda::DeviceBuffer<T>,
    expected: usize,
    operand: &'static str,
) -> Result<()> {
    if buffer.len() == expected {
        Ok(())
    } else {
        Err(Error::InvalidDecoderKernel(operand))
    }
}

/// The number of `width`-wide rows in `elements`, refusing an empty width
/// or a partial row.
fn rows_of(elements: usize, width: usize, operand: &'static str) -> Result<usize> {
    if width == 0 || !elements.is_multiple_of(width) {
        return Err(Error::InvalidDecoderKernel(operand));
    }
    Ok(elements / width)
}
