use mircuda::{
    CublasGemmOffsets, CublasGemmOperand, CublasGemmSpec, DeviceBuffer, PaddedAttentionPlan,
};

use super::{
    device::Device,
    plans::{Input, Operands},
};
use crate::{
    Result,
    kernels::{DecisionWindow, HeadLayout, RopeTables, ScoreRows},
};

/// Queries per batch member of a banded product.
pub const BLOCK: usize = 64;

/// Banded attention of one layer over a fused `[tokens, 3 × heads × head_dim]`
/// projection of sequences padded to `length`: keys within `radius` tokens.
#[derive(Clone, Copy)]
pub struct Band {
    pub length: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub radius: usize,
}

/// Length a batch is padded to: a banded layer multiplies `BLOCK`-query
/// blocks against their windows only when every sequence spans whole blocks.
/// Unwindowed attention of every token, fused so scores never reach memory,
/// over a token-major projection whose queries and keys are already rotated.
pub fn attend_full<I: Input>(
    device: &Device<'_>,
    (qkv, lengths): (&DeviceBuffer<I>, &DeviceBuffer<u32>),
    (length, heads, dim): (usize, usize, usize),
) -> Result<DeviceBuffer<I>> {
    let mut output = device.buffer::<I>(qkv.len() / 3)?;
    let scale = 1.0 / f32::from(u16::try_from(dim)?).sqrt();
    let context = &device.backend.inner.context;
    PaddedAttentionPlan::<I>::new(context, device.stream(), heads, dim)?.execute(
        device.stream(),
        (qkv, lengths),
        length,
        scale,
        &mut output,
    )?;
    Ok(output)
}

pub const fn padded_length(length: usize, radius: Option<usize>) -> usize {
    match radius {
        Some(radius) if length > BLOCK + 2 * radius => length.div_ceil(BLOCK) * BLOCK,
        _ => length,
    }
}

impl Band {
    /// How queries group into batch members: blocks against their window of
    /// keys, or whole short sequences under the band mask.
    fn rows(self, lengths: &DeviceBuffer<u32>, tokens: usize) -> ScoreRows<'_> {
        let whole = ScoreRows {
            lengths,
            queries: self.length,
            keys: self.length,
            groups: lengths.len(),
            shift: 0,
            length: self.length,
            window: DecisionWindow::Band { radius: self.radius },
        };
        if self.length.is_multiple_of(BLOCK) && self.length > BLOCK + 2 * self.radius {
            ScoreRows {
                queries: BLOCK,
                keys: BLOCK + 2 * self.radius,
                groups: tokens / BLOCK,
                shift: self.radius,
                ..whole
            }
        } else {
            whole
        }
    }
}

/// Banded attention of every token through strided batched products over
/// head-major operands, token-major and ready for the output projection;
/// rows of padded queries are zeros.
pub fn attend_band<I: Input>(
    device: &Device<'_>,
    (qkv, lengths): (&DeviceBuffer<f32>, &DeviceBuffer<u32>),
    rope: &RopeTables,
    band: Band,
) -> Result<DeviceBuffer<I>> {
    let (heads, dim) = (band.heads, band.head_dim);
    let tokens = qkv.len() / (3 * heads * dim);
    let rows = band.rows(lengths, tokens);
    let layout = HeadLayout {
        tokens,
        length: band.length,
        heads,
        dim,
        pad: rows.shift,
    };
    let mut operands = device.buffer::<I>(layout.elements())?;
    I::split(device.layout, device.stream(), qkv, rope, layout, &mut operands)?;
    let members = heads * rows.groups;
    let mut scores = device.buffer::<f32>(members * rows.queries * rows.keys)?;
    let member = |leading, stride, transposed| CublasGemmOperand { leading, stride, transposed };
    let block = rows.queries * dim;
    let weights = member(rows.keys, rows.queries * rows.keys, false);
    let scale = 1.0 / f32::from(u16::try_from(dim)?).sqrt();
    let spec = CublasGemmSpec::new(
        (rows.queries, rows.keys, dim, members),
        member(dim, block, false),
        member(dim, block, true),
        weights,
    )?;
    let window = rows.shift * dim;
    let offsets = CublasGemmOffsets {
        left: layout.part(0),
        right: layout.part(1) - window,
        output: 0,
    };
    let products = Operands {
        left: &operands,
        right: &operands,
        output: &mut scores,
        offsets,
    };
    device.plans.multiply(device.backend, spec, products, (scale, 0.0))?;
    let mut probabilities = device.buffer::<I>(scores.len())?;
    I::softmax(device.layout, device.stream(), &scores, &rows, &mut probabilities)?;
    let mut mixed = device.buffer::<f32>(heads * tokens * dim)?;
    let spec = CublasGemmSpec::new(
        (rows.queries, dim, rows.keys, members),
        weights,
        member(dim, block, false),
        member(dim, block, false),
    )?;
    let offsets = CublasGemmOffsets {
        left: 0,
        right: layout.part(2) - window,
        output: 0,
    };
    let products = Operands {
        left: &probabilities,
        right: &operands,
        output: &mut mixed,
        offsets,
    };
    device.plans.multiply(device.backend, spec, products, (1.0, 0.0))?;
    let mut merged = device.buffer::<I>(mixed.len())?;
    I::merge(device.layout, device.stream(), &mixed, (heads, dim), &mut merged)?;
    Ok(merged)
}
