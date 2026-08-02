use models::weights::{
    BlockProjectionLayout, BlockQuantization, TensorBinding, TensorPacking, TensorStorage,
};

use crate::engine::{Array, Dtype, Error, ModelTensors, Result, Stream};

#[derive(Debug)]
pub(in crate::engine) struct MxFp8Linear {
    arrays: MxFp8Arrays,
    bias: Option<Array>,
    layout: MxFp8Layout,
}

#[derive(Clone, Copy, Debug)]
enum MxFp8Layout {
    Matrix,
    Gathered,
}

#[derive(Debug)]
pub(in crate::engine) struct MxFp8Embedding {
    arrays: MxFp8Arrays,
}

#[derive(Debug)]
struct MxFp8Arrays {
    weight: Array,
    scales: Array,
}

pub(super) fn linear(tensors: &ModelTensors, binding: &TensorBinding) -> Result<MxFp8Linear> {
    let TensorStorage::BlockQuantized {
        format,
        scales,
        global_scale: None,
        input_scale: None,
        bias,
        packing: _,
    } = &binding.storage
    else {
        return Err(invalid(binding, "requires separate MXFP8 storage without higher scales"));
    };
    require_native(binding, *format)?;
    let (layout, prefix, output, input) = projection_shape(binding)?;
    let arrays = load_arrays(tensors, binding, scales, &prefix, output, input)?;
    let bias = if let Some(name) = bias {
        let bias = tensors.get(name)?;
        let mut shape = prefix;
        shape.push(output);
        require(&bias, Dtype::Bfloat16, &shape, binding, "bias")?;
        Some(bias)
    } else {
        None
    };
    Ok(MxFp8Linear { arrays, bias, layout })
}

pub(super) fn embedding(tensors: &ModelTensors, binding: &TensorBinding) -> Result<MxFp8Embedding> {
    let TensorStorage::BlockQuantized {
        format,
        scales,
        global_scale: None,
        input_scale: None,
        bias: None,
        packing: TensorPacking::Separate,
    } = &binding.storage
    else {
        return Err(invalid(binding, "requires separate MXFP8 embedding storage"));
    };
    require_native(binding, *format)?;
    let (layout, prefix, output, input) = projection_shape(binding)?;
    if !matches!(layout, MxFp8Layout::Matrix) {
        return Err(invalid(binding, "MXFP8 embedding must be an ordinary matrix"));
    }
    Ok(MxFp8Embedding {
        arrays: load_arrays(tensors, binding, scales, &prefix, output, input)?,
    })
}

impl MxFp8Linear {
    pub(super) fn forward(&self, input: &Array, stream: &Stream) -> Result<Array> {
        if !matches!(self.layout, MxFp8Layout::Matrix) {
            return Err(Error::InvalidQuantization(
                "gathered MXFP8 matrix bank does not support ordinary execution".into(),
            ));
        }
        let output =
            stream
                .native()
                .graph()
                .mxfp8_matmul(input.native(), self.arrays.native(), true)?;
        let output = Array::from_native(output)?.astype_like(input, stream)?;
        if let Some(bias) = &self.bias {
            output.add(bias, stream)
        } else {
            Ok(output)
        }
    }

    pub(super) fn gather(&self, input: &Array, indices: &Array, stream: &Stream) -> Result<Array> {
        let MxFp8Layout::Gathered = self.layout else {
            return Err(Error::InvalidQuantization(
                "ordinary MXFP8 matrix does not support gathered execution".into(),
            ));
        };
        if input.dtype()? != Dtype::Bfloat16 || indices.dtype()? != Dtype::Uint32 {
            return Err(Error::InvalidQuantization(
                "gathered MXFP8 requires BF16 input and U32 indices".into(),
            ));
        }
        let graph = stream.native().graph();
        let weight =
            Array::from_native(graph.take(self.arrays.weight.native(), indices.native(), 0)?)?;
        let scales =
            Array::from_native(graph.take(self.arrays.scales.native(), indices.native(), 0)?)?;
        let selected = MxFp8Arrays { weight, scales };
        let output =
            Array::from_native(graph.mxfp8_matmul(input.native(), selected.native(), true)?)?
                .astype_like(input, stream)?;
        if let Some(bias) = &self.bias {
            let bias = graph.take(bias.native(), indices.native(), 0)?;
            let bias = graph.expand_dims(&bias, &[-2])?;
            Array::from_native(graph.add(output.native(), &bias)?)
        } else {
            Ok(output)
        }
    }

    pub(super) const fn has_bias(&self) -> bool {
        self.bias.is_some()
    }
}

impl MxFp8Embedding {
    pub(super) fn lookup(&self, indices: &Array, stream: &Stream) -> Result<Array> {
        let graph = stream.native().graph();
        let weight = graph.take(self.arrays.weight.native(), indices.native(), 0)?;
        let scales = graph.take(self.arrays.scales.native(), indices.native(), 0)?;
        Array::from_native(
            graph.dequantize_mxfp8(mirtal::MxFp8 { weight: &weight, scales: &scales })?,
        )
    }

    pub(super) fn project(&self, input: &Array, stream: &Stream) -> Result<Array> {
        let output =
            stream
                .native()
                .graph()
                .mxfp8_matmul(input.native(), self.arrays.native(), true)?;
        Array::from_native(output)?.astype_like(input, stream)
    }
}

impl MxFp8Arrays {
    const fn native(&self) -> mirtal::MxFp8<'_> {
        mirtal::MxFp8 {
            weight: self.weight.native(),
            scales: self.scales.native(),
        }
    }
}

fn load_arrays(
    tensors: &ModelTensors,
    binding: &TensorBinding,
    scales: &str,
    prefix: &[usize],
    output: usize,
    input: usize,
) -> Result<MxFp8Arrays> {
    let mut weight_shape = prefix.to_vec();
    weight_shape.extend([output, input / 4]);
    let weight = tensors.get(&binding.source)?;
    require(&weight, Dtype::Uint32, &weight_shape, binding, "weight")?;
    let mut scale_shape = prefix.to_vec();
    scale_shape.extend([output, input / 32]);
    let scales = tensors.get(scales)?;
    require(&scales, Dtype::Uint8, &scale_shape, binding, "scales")?;
    Ok(MxFp8Arrays { weight, scales })
}

fn require_native(binding: &TensorBinding, format: BlockQuantization) -> Result<()> {
    if format == BlockQuantization::MXFP8 && binding.block_projection_layout().is_some() {
        Ok(())
    } else {
        Err(invalid(binding, "does not match the native Metal MXFP8 contract"))
    }
}

fn projection_shape(binding: &TensorBinding) -> Result<(MxFp8Layout, Vec<usize>, usize, usize)> {
    let (layout, prefix, output, input) =
        match (binding.block_projection_layout(), binding.logical_shape.as_deref()) {
            (Some(BlockProjectionLayout::Matrix), Some([output, input])) => {
                (MxFp8Layout::Matrix, Vec::new(), *output, *input)
            },
            (
                Some(BlockProjectionLayout::MatrixBank { matrices }),
                Some([actual, output, input]),
            ) if matrices == *actual => (MxFp8Layout::Gathered, vec![matrices], *output, *input),
            (
                Some(BlockProjectionLayout::FusedGateUpBank { experts, interleaved: true }),
                Some([actual, output, input]),
            ) if experts == *actual => (MxFp8Layout::Gathered, vec![experts], *output, *input),
            _ => return Err(invalid(binding, "requires an ordinary or gathered matrix layout")),
        };
    if !input.is_multiple_of(BlockQuantization::MXFP8.block_size) {
        return Err(invalid(binding, "input width is not a complete MXFP8 block"));
    }
    Ok((layout, prefix, output, input))
}

fn require(
    array: &Array,
    dtype: Dtype,
    shape: &[usize],
    binding: &TensorBinding,
    kind: &str,
) -> Result<()> {
    let expected = shape
        .iter()
        .copied()
        .map(i32::try_from)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if array.dtype()? == dtype && array.shape()? == expected {
        Ok(())
    } else {
        Err(invalid(binding, &format!("{kind} dtype or shape differs from the contract")))
    }
}

fn invalid(binding: &TensorBinding, reason: &str) -> Error {
    Error::InvalidQuantization(format!("{}: {reason}", binding.source))
}
