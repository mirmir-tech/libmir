use mircuda::DeviceBuffer;
use models::decision::DecisionRow;

use super::{attention::padded_length, staging::Staging};
use crate::{CudaBackend, Error, Result};

/// Decision rows padded to one length and copied to the device.
pub struct DeviceBatch {
    pub rows: usize,
    pub length: usize,
    pub tokens: DeviceBuffer<u32>,
    pub lengths: DeviceBuffer<u32>,
    pub kinds: DeviceBuffer<u32>,
    /// Flat index of every option marker, row by row.
    pub markers: DeviceBuffer<u32>,
    /// Markers of each row, in row order.
    pub marker_counts: Vec<usize>,
}

impl DeviceBatch {
    /// `band` is the widest local attention band, which may pad the rows to
    /// whole attention blocks.
    pub fn new(
        (backend, staging): (&CudaBackend, &mut Staging),
        rows: &[DecisionRow],
        positions: usize,
        band: Option<usize>,
    ) -> Result<Self> {
        let longest = rows
            .iter()
            .map(|row| row.tokens.len())
            .max()
            .ok_or(Error::InvalidDecoderKernel("decision batch is empty"))?;
        if longest > positions {
            return Err(Error::InvalidDecoderKernel("decision row exceeds the position budget"));
        }
        let length = padded_length(longest, band);
        let mut tokens = Vec::with_capacity(rows.len() * length);
        let mut markers = Vec::new();
        for (index, row) in rows.iter().enumerate() {
            tokens.extend(&row.tokens);
            tokens.resize(tokens.len() + length - row.tokens.len(), 0);
            for &marker in &row.markers {
                markers.push(u32::try_from(index * length + marker)?);
            }
        }
        let lengths = rows
            .iter()
            .map(|row| u32::try_from(row.tokens.len()))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let kinds = rows
            .iter()
            .map(|row| u32::try_from(row.kind.index()))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let [tokens, lengths, kinds, markers]: [DeviceBuffer<u32>; 4] = staging
            .upload(backend, &[&tokens, &lengths, &kinds, &markers])?
            .try_into()
            .map_err(|_| Error::InvalidDecoderKernel("decision batch upload"))?;
        Ok(Self {
            rows: rows.len(),
            length,
            tokens,
            lengths,
            kinds,
            markers,
            marker_counts: rows.iter().map(|row| row.markers.len()).collect(),
        })
    }

    pub const fn tokens(&self) -> usize {
        self.rows * self.length
    }
}
