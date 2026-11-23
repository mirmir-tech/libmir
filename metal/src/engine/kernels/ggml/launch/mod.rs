mod routed;

pub(super) use routed::{gathered, grouped};

use super::{GGML_GEMV_MAX_TOKENS, GgmlMatrix};
use crate::engine::{Array, Error, Result, Stream, kernels::template};

/// Rows per simdgroup and simdgroups per threadgroup of the GEMV.
const ROWS: usize = 1;
pub(super) const SIMDGROUPS: usize = 4;
/// Tokens the GEMV serves: it decodes each weight sub-block once for all of
/// them, which beats the 8-token simdgroup-matrix tile for two tokens
/// (`Q4_K` 4096 to 12288: 128 vs 162 us) but not for three or more.
const GEMV_TOKENS: usize = 2;
/// Tokens of one simdgroup-matrix launch: two 8-token tiles; more spill
/// registers, so wider batches run in chunks.
const MMA_TOKENS: usize = 16;

pub(super) fn gemv(
    [kernel, mma]: [&mirtal::MetalKernel<2, 1>; 2],
    stream: &Stream,
    input: &Array,
    weight: &Array,
    matrix: GgmlMatrix,
) -> Result<Array> {
    let (shape, tokens) = output_shape(input, matrix.columns, matrix.rows)?;
    if tokens == 0 || tokens > GGML_GEMV_MAX_TOKENS {
        return Err(Error::InvalidQuantization(format!(
            "GGML GEMV serves 1 to {GGML_GEMV_MAX_TOKENS} tokens, not {tokens}"
        )));
    }
    let dtype = input.native().dtype()?;
    if tokens > MMA_TOKENS {
        let rows = i32::try_from(tokens)?;
        let flat = input.reshape(&[rows, i32::try_from(matrix.columns)?], stream)?;
        let chunks = (0..tokens)
            .step_by(MMA_TOKENS)
            .map(|start| {
                let stop = (start + MMA_TOKENS).min(tokens);
                let chunk = flat.slice(&[start, 0], &[stop, matrix.columns], stream)?;
                gemv([kernel, mma], stream, &chunk, weight, matrix)
            })
            .collect::<Result<Vec<_>>>()?;
        let joined = Array::concatenate(&chunks.iter().collect::<Vec<_>>(), 0, stream)?;
        let shape = shape
            .iter()
            .map(|&value| i32::try_from(value))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        return joined.reshape(&shape, stream);
    }
    let spec = [mirtal::OutputSpec::new(mirtal::Shape::new(shape)?, dtype)];
    if tokens > GEMV_TOKENS {
        // Two simdgroups of two 8-row tiles; see `kernels/ggml/mma.metal`.
        let [output] = mma.dispatch(
            stream.native(),
            [input.native(), weight.native()],
            &spec,
            &mirtal::Dispatch::new([matrix.rows.div_ceil(32) * 64, 1, 1], [64, 1, 1]).templates([
                mirtal::TemplateArg::dtype("T", dtype),
                type_template(matrix)?,
                template("INPUT", matrix.columns)?,
                template("OUTPUT", matrix.rows)?,
                template("TOKENS", tokens)?,
            ]),
        )?;
        return Array::from_native(output);
    }
    let threads = 32 * SIMDGROUPS;
    let groups = matrix.rows.div_ceil(ROWS * SIMDGROUPS);
    let [output] = kernel.dispatch(
        stream.native(),
        [input.native(), weight.native()],
        &spec,
        &mirtal::Dispatch::new([groups * threads, 1, 1], [threads, 1, 1]).templates([
            mirtal::TemplateArg::dtype("T", dtype),
            type_template(matrix)?,
            template("INPUT", matrix.columns)?,
            template("OUTPUT", matrix.rows)?,
            template("TOKENS", tokens)?,
            template("ROWS", ROWS)?,
        ]),
    )?;
    Array::from_native(output)
}

pub(super) fn dequantize(
    kernel: &mirtal::MetalKernel<1, 1>,
    stream: &Stream,
    weight: &Array,
    matrix: GgmlMatrix,
    dtype: mirtal::DType,
) -> Result<Array> {
    let subs = matrix.columns / 32;
    let [output] = kernel.dispatch(
        stream.native(),
        [weight.native()],
        &[mirtal::OutputSpec::new(mirtal::Shape::new([matrix.rows, matrix.columns])?, dtype)],
        &mirtal::Dispatch::new([subs, matrix.rows, 1], [subs.min(64), 1, 1]).templates([
            mirtal::TemplateArg::dtype("T", dtype),
            type_template(matrix)?,
            template("INPUT", matrix.columns)?,
        ]),
    )?;
    Array::from_native(output)
}

pub(super) fn embedding(
    kernel: &mirtal::MetalKernel<2, 1>,
    stream: &Stream,
    weight: &Array,
    indices: &Array,
    matrix: GgmlMatrix,
    dtype: mirtal::DType,
) -> Result<Array> {
    if indices.native().dtype()? != mirtal::DType::Uint32 {
        return Err(Error::InvalidQuantization("GGML embedding indices must be U32".into()));
    }
    let mut shape = indices
        .shape()?
        .into_iter()
        .map(usize::try_from)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let tokens = shape
        .iter()
        .try_fold(1_usize, |total, value| total.checked_mul(*value).ok_or(Error::ShapeOverflow))?;
    shape.push(matrix.columns);
    let subs = matrix.columns / 32;
    let [output] = kernel.dispatch(
        stream.native(),
        [weight.native(), indices.native()],
        &[mirtal::OutputSpec::new(mirtal::Shape::new(shape)?, dtype)],
        &mirtal::Dispatch::new([subs, tokens, 1], [subs.min(64), 1, 1]).templates([
            mirtal::TemplateArg::dtype("T", dtype),
            type_template(matrix)?,
            template("HIDDEN", matrix.columns)?,
            template("VOCABULARY", matrix.rows)?,
        ]),
    )?;
    Array::from_native(output)
}

pub(super) fn mxfp4_repack(
    kernel: &mirtal::MetalKernel<1, 2>,
    stream: &Stream,
    weight: &Array,
) -> Result<[Array; 2]> {
    let shape = weight.shape()?;
    let (rows, bytes) = match *shape.as_slice() {
        [rows, bytes] if bytes % 17 == 0 => (usize::try_from(rows)?, usize::try_from(bytes)?),
        _ => {
            return Err(Error::InvalidQuantization(format!(
                "GGML MXFP4 rows {shape:?} are not whole 17-byte blocks"
            )));
        },
    };
    let blocks = bytes / 17;
    let outputs = [
        mirtal::OutputSpec::new(mirtal::Shape::new([rows, blocks, 16])?, mirtal::DType::Uint8),
        mirtal::OutputSpec::new(mirtal::Shape::new([rows, blocks])?, mirtal::DType::Uint8),
    ];
    let [blocks_out, scales] = kernel.dispatch(
        stream.native(),
        [weight.native()],
        &outputs,
        &mirtal::Dispatch::new([blocks, rows, 1], [blocks.min(64), 1, 1])
            .templates([template("BLOCKS", blocks)?]),
    )?;
    Ok([Array::from_native(blocks_out)?, Array::from_native(scales)?])
}

pub(in crate::engine::kernels::ggml) fn type_template(
    matrix: GgmlMatrix,
) -> Result<mirtal::TemplateArg> {
    template("TYPE", usize::try_from(matrix.ggml_type.id())?)
}

/// `input.shape` with its last dimension `columns` replaced by `rows`, and
/// the number of input rows.
pub(super) fn output_shape(
    input: &Array,
    columns: usize,
    rows: usize,
) -> Result<(Vec<usize>, usize)> {
    let mut shape = input
        .shape()?
        .into_iter()
        .map(usize::try_from)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    match shape.last_mut() {
        Some(width) if *width == columns => *width = rows,
        _ => {
            return Err(Error::InvalidQuantization(format!(
                "GGML input {shape:?} does not end in {columns} columns"
            )));
        },
    }
    let tokens = shape[..shape.len() - 1]
        .iter()
        .try_fold(1_usize, |total, value| total.checked_mul(*value).ok_or(Error::ShapeOverflow))?;
    Ok((shape, tokens))
}
