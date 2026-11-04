use mircuda::{CublasElement, DeviceBuffer, LaunchConfig, PaddedAttentionElement, Stream, bf16};

use super::{
    DecisionLayout, HeadLayout, RopeTables, ScoreRows, elements_launch, require, rows_launch,
};
use crate::{Error, Result};

/// Dynamic shared memory a block may use without opting in.
const SOFTMAX_SHARED_BYTES: usize = 48 * 1024;

/// Element type an activation feeding a matrix product is written in.
pub trait DecisionElement: CublasElement + PaddedAttentionElement {
    /// Layer normalisation of every `weight.len()`-wide row.
    fn norm(
        layout: &DecisionLayout,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        norm: (&DeviceBuffer<f32>, &DeviceBuffer<f32>, f32),
        output: &mut DeviceBuffer<Self>,
    ) -> Result<()>;

    /// `gelu(x) · gate` of `[rows, 2 × width]` halves.
    fn geglu(
        layout: &DecisionLayout,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<Self>,
        width: usize,
    ) -> Result<()>;

    /// A token-major fused projection to head-major operands with rotated
    /// queries and keys.
    fn split(
        layout: &DecisionLayout,
        stream: &Stream,
        qkv: &DeviceBuffer<f32>,
        rope: &RopeTables,
        heads: HeadLayout,
        output: &mut DeviceBuffer<Self>,
    ) -> Result<()>;

    /// A token-major fused projection with rotated queries and keys.
    fn rotate(
        layout: &DecisionLayout,
        stream: &Stream,
        qkv: &DeviceBuffer<f32>,
        rope: &RopeTables,
        shape: (usize, usize, usize),
        output: &mut DeviceBuffer<Self>,
    ) -> Result<()>;

    /// `[heads][tokens][dim]` outputs to token-major rows.
    fn merge(
        layout: &DecisionLayout,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        shape: (usize, usize),
        output: &mut DeviceBuffer<Self>,
    ) -> Result<()>;

    /// Attention weights of every score row.
    fn softmax(
        layout: &DecisionLayout,
        stream: &Stream,
        scores: &DeviceBuffer<f32>,
        rows: &ScoreRows<'_>,
        output: &mut DeviceBuffer<Self>,
    ) -> Result<()>;
}

macro_rules! decision_element {
    ($element:ty, $variant:tt) => {
        impl DecisionElement for $element {
            fn norm(
                layout: &DecisionLayout,
                stream: &Stream,
                input: &DeviceBuffer<f32>,
                (weight, bias, epsilon): (&DeviceBuffer<f32>, &DeviceBuffer<f32>, f32),
                output: &mut DeviceBuffer<Self>,
            ) -> Result<()> {
                require(output, input.len(), "decision norm output")?;
                let rows = input.len() / weight.len();
                let geometry = (u32::try_from(rows)?, u32::try_from(weight.len())?);
                Ok(layout.norm.$variant.launch(
                    stream,
                    rows_launch(rows)?,
                    (input, weight, bias, output, geometry.0, geometry.1, epsilon),
                )?)
            }

            fn geglu(
                layout: &DecisionLayout,
                stream: &Stream,
                input: &DeviceBuffer<f32>,
                output: &mut DeviceBuffer<Self>,
                width: usize,
            ) -> Result<()> {
                let rows = output.len() / width;
                require(input, rows * 2 * width, "decision GeGLU input")?;
                let geometry = (u32::try_from(rows)?, u32::try_from(width)?);
                Ok(layout.geglu.$variant.launch(
                    stream,
                    elements_launch(output.len())?,
                    (input, output, geometry.0, geometry.1),
                )?)
            }

            fn split(
                layout: &DecisionLayout,
                stream: &Stream,
                qkv: &DeviceBuffer<f32>,
                rope: &RopeTables,
                heads: HeadLayout,
                output: &mut DeviceBuffer<Self>,
            ) -> Result<()> {
                require(qkv, heads.tokens * 3 * heads.heads * heads.dim, "decision split input")?;
                require(output, heads.elements(), "decision split output")?;
                check_rope(rope, heads.length, heads.dim)?;
                let arguments = (
                    u32::try_from(heads.tokens)?,
                    u32::try_from(heads.length)?,
                    u32::try_from(heads.heads)?,
                    u32::try_from(heads.dim)?,
                    u32::try_from(heads.pad)?,
                );
                Ok(layout.split_rope.$variant.launch(
                    stream,
                    elements_launch(heads.elements() / 2)?,
                    (
                        qkv, &rope.cosines, &rope.sines, output, arguments.0, arguments.1,
                        arguments.2, arguments.3, arguments.4,
                    ),
                )?)
            }

            fn rotate(
                layout: &DecisionLayout,
                stream: &Stream,
                qkv: &DeviceBuffer<f32>,
                rope: &RopeTables,
                (length, heads, dim): (usize, usize, usize),
                output: &mut DeviceBuffer<Self>,
            ) -> Result<()> {
                require(output, qkv.len(), "decision rotate output")?;
                check_rope(rope, length, dim)?;
                let arguments = (
                    u32::try_from(qkv.len() / (3 * heads * dim))?,
                    u32::try_from(length)?,
                    u32::try_from(heads)?,
                    u32::try_from(dim)?,
                );
                Ok(layout.rotate.$variant.launch(
                    stream,
                    elements_launch(qkv.len() / 2)?,
                    (
                        qkv, &rope.cosines, &rope.sines, output, arguments.0, arguments.1,
                        arguments.2, arguments.3,
                    ),
                )?)
            }

            fn merge(
                layout: &DecisionLayout,
                stream: &Stream,
                input: &DeviceBuffer<f32>,
                (heads, dim): (usize, usize),
                output: &mut DeviceBuffer<Self>,
            ) -> Result<()> {
                require(output, input.len(), "decision merge output")?;
                let tokens = u32::try_from(input.len() / (heads * dim))?;
                let geometry = (u32::try_from(heads)?, u32::try_from(dim)?);
                Ok(layout.merge.$variant.launch(
                    stream,
                    elements_launch(input.len())?,
                    (input, output, tokens, geometry.0, geometry.1),
                )?)
            }

            fn softmax(
                layout: &DecisionLayout,
                stream: &Stream,
                scores: &DeviceBuffer<f32>,
                rows: &ScoreRows<'_>,
                output: &mut DeviceBuffer<Self>,
            ) -> Result<()> {
                require(output, scores.len(), "decision softmax output")?;
                let cached = rows.keys * size_of::<f32>();
                if cached > SOFTMAX_SHARED_BYTES {
                    return Err(Error::InvalidDecoderKernel("decision softmax row length"));
                }
                let launch = LaunchConfig {
                    shared_memory_bytes: u32::try_from(cached)?,
                    ..rows_launch(scores.len() / rows.keys)?
                };
                let arguments = (
                    u32::try_from(rows.queries)?,
                    u32::try_from(rows.keys)?,
                    u32::try_from(rows.groups)?,
                    u32::try_from(rows.shift)?,
                    u32::try_from(rows.length)?,
                    rows.window.kernel_radius()?,
                );
                Ok(layout.softmax.$variant.launch(
                    stream,
                    launch,
                    (
                        scores, output, rows.lengths, arguments.0, arguments.1, arguments.2,
                        arguments.3, arguments.4, arguments.5,
                    ),
                )?)
            }
        }
    };
}

fn check_rope(rope: &RopeTables, length: usize, dim: usize) -> Result<()> {
    if rope.cosines.len() < length * dim / 2 || rope.sines.len() != rope.cosines.len() {
        return Err(Error::InvalidDecoderKernel("decision rope table length"));
    }
    Ok(())
}

decision_element!(f32, 0);
decision_element!(bf16, 1);
