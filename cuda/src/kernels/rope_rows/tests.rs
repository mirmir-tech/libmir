//! Launch validation of the rotary row restoration.

use super::RopeRows;
use crate::{CudaBackend, CudaConfig, Error, Result};

const HEAD_DIM: usize = 8;
const ROTARY: usize = 4;

/// The launch's outcome over `rows` rows of `bytes` bytes in all.
fn launch(rows: usize, bytes: usize) -> Result<Result<()>> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let operation = RopeRows::compile(backend.compiler())?;
    let source = backend.pool().allocate_zeroed::<u8>(backend.stream(), bytes)?;
    let mut output = backend.pool().allocate_zeroed::<u8>(backend.stream(), bytes)?;
    Ok(operation.execute(backend.stream(), &source, &mut output, rows, (HEAD_DIM, ROTARY)))
}

/// Zero rows are a whole number of heads but have no row width to divide
/// out; they are refused rather than divided by.
#[test]
fn refuses_zero_rows() -> Result<()> {
    assert!(matches!(launch(0, 64)?, Err(Error::InvalidDecoderKernel(_))));
    Ok(())
}

/// Fewer bytes than rows leave rows of no bytes.
#[test]
fn refuses_rows_without_bytes() -> Result<()> {
    assert!(matches!(launch(HEAD_DIM, HEAD_DIM / 2)?, Err(Error::InvalidDecoderKernel(_))));
    Ok(())
}

/// Rows that are not whole heads are refused as before, and whole heads
/// still launch.
#[test]
fn refuses_partial_heads_only() -> Result<()> {
    let partial = launch(HEAD_DIM + 1, 4 * (HEAD_DIM + 1))?;
    assert!(matches!(partial, Err(Error::InvalidDecoderKernel(_))));
    launch(2 * HEAD_DIM, 4 * 2 * HEAD_DIM)?
}
