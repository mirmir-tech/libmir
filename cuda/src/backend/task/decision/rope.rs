use super::weights::upload;
use crate::{CudaBackend, Result, kernels::RopeTables};

/// Rotate-half angles of `positions` tokens, rounded like `PyTorch`'s f32
/// rotary embedding: f32 frequencies and angles, correctly rounded cosines.
pub fn tables(
    backend: &CudaBackend,
    theta: f32,
    positions: usize,
    head_dim: usize,
) -> Result<RopeTables> {
    let dim = f32::from(u16::try_from(head_dim)?);
    let frequencies = (0..head_dim / 2)
        .map(|pair| Ok(1.0 / theta.powf(f32::from(u16::try_from(2 * pair)?) / dim)))
        .collect::<Result<Vec<f32>>>()?;
    let mut cosines = Vec::with_capacity(positions * frequencies.len());
    let mut sines = Vec::with_capacity(positions * frequencies.len());
    for position in 0..positions {
        let position = f32::from(u16::try_from(position)?);
        for frequency in &frequencies {
            let angle = position * frequency;
            cosines.push(angle.cos());
            sines.push(angle.sin());
        }
    }
    Ok(RopeTables {
        cosines: upload(backend, &cosines)?,
        sines: upload(backend, &sines)?,
    })
}
