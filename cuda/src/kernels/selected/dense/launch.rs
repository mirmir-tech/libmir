use mircuda::LaunchConfig;

use super::{DenseGateUpLayout, SelectedDenseMoeSpec};
use crate::{Result, kernels::geometry::narrow};

pub(super) fn gated(spec: SelectedDenseMoeSpec) -> Result<LaunchConfig> {
    let tiled = spec.gate_transposed
        && spec.up_transposed
        && spec.gate_up_layout == DenseGateUpLayout::FusedInterleaved;
    let (rows, block, shared_memory_bytes) = if tiled {
        (32, (32, 2, 1), 4 * 32 * size_of::<f32>())
    } else {
        (8, (32, 8, 1), 0)
    };
    Ok(LaunchConfig {
        grid: (
            narrow(spec.output_features.div_ceil(rows))?,
            narrow(spec.selected_count)?,
            narrow(spec.tokens)?,
        ),
        block,
        shared_memory_bytes: narrow(shared_memory_bytes)?,
    })
}

pub(super) fn reduce(spec: SelectedDenseMoeSpec) -> Result<LaunchConfig> {
    let tiled = spec.down_transposed;
    let (rows, shared_memory_bytes) = if tiled {
        (32, 8 * 32 * size_of::<f32>())
    } else {
        (8, 0)
    };
    Ok(LaunchConfig {
        grid: (narrow(spec.input_features.div_ceil(rows))?, narrow(spec.tokens)?, 1),
        block: (32, 8, 1),
        shared_memory_bytes: narrow(shared_memory_bytes)?,
    })
}

pub(super) fn project(spec: SelectedDenseMoeSpec) -> Result<LaunchConfig> {
    if !spec.down_transposed {
        return Ok(LaunchConfig {
            grid: (
                narrow(spec.input_features.div_ceil(8))?,
                narrow(spec.selected_count)?,
                narrow(spec.tokens)?,
            ),
            block: (32, 8, 1),
            shared_memory_bytes: 0,
        });
    }
    let rows = if spec.input_features.is_multiple_of(2) {
        64
    } else {
        32
    };
    let items = rows / 32;
    Ok(LaunchConfig {
        grid: (
            narrow(spec.input_features.div_ceil(rows))?,
            narrow(spec.selected_count)?,
            narrow(spec.tokens)?,
        ),
        block: (32, 8, 1),
        shared_memory_bytes: narrow(8 * items * 32 * size_of::<f32>())?,
    })
}

pub(super) fn finalize(spec: SelectedDenseMoeSpec) -> Result<LaunchConfig> {
    Ok(LaunchConfig {
        grid: (narrow(spec.input_features.div_ceil(256))?, narrow(spec.tokens)?, 1),
        block: (256, 1, 1),
        shared_memory_bytes: 0,
    })
}
