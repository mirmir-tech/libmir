use models::weights::{
    BlockProjectionLayout, BlockQuantization, BlockStorageDType, TensorBinding, TensorStorage,
};

use crate::engine::{Array, Dtype, Error, ModelTensors, Result, Stream};

#[cfg(test)]
mod accuracy;
mod execution;
#[cfg(test)]
mod experts;
mod fusion;
#[cfg(test)]
mod joined;

#[derive(Debug)]
pub(in crate::engine) struct MxFp4Linear {
    pub(in crate::engine) weight: Array,
    pub(in crate::engine) scales: Array,
    pub(in crate::engine) bias: Array,
    input_features: usize,
    output_features: usize,
    pub(in crate::engine) has_bias: bool,
    pub(in crate::engine) layout: MxFp4LinearLayout,
}

#[derive(Debug, Clone, Copy)]
pub(in crate::engine) enum MxFp4LinearLayout {
    Matrix,
    Gathered,
}

#[derive(Debug)]
pub(in crate::engine) struct MxFp4Embedding {
    linear: MxFp4Linear,
}

pub(super) fn linear(
    tensors: &ModelTensors,
    binding: &TensorBinding,
    stream: &Stream,
) -> Result<MxFp4Linear> {
    let TensorStorage::BlockQuantized {
        format,
        scales,
        global_scale: None,
        input_scale: None,
        bias,
        packing: _,
    } = &binding.storage
    else {
        return Err(invalid(binding, "requires self-contained MXFP4 storage"));
    };
    if !format.is_mxfp4() {
        return Err(invalid(binding, "does not match the Metal MXFP4 contract"));
    }
    let (layout, prefix, output_features, input_features) = projection_shape(binding)?;
    let (weight_dtype, tail) = match format.storage_dtype {
        BlockStorageDType::U8 => (Dtype::Uint8, vec![input_features / 32, 16]),
        BlockStorageDType::U32 => (Dtype::Uint32, vec![input_features / 8]),
        _ => return Err(invalid(binding, "uses an unsupported MXFP4 container dtype")),
    };
    let mut weight_shape = prefix.clone();
    weight_shape.push(output_features);
    weight_shape.extend(tail);
    let weight = tensors.get(&binding.source)?;
    require(&weight, weight_dtype, &weight_shape, binding, "weight")?;
    let mut scale_shape = prefix.clone();
    scale_shape.extend([output_features, input_features / 32]);
    let scales = tensors.get(scales)?;
    require(&scales, Dtype::Uint8, &scale_shape, binding, "scales")?;
    let mut bias_shape = prefix;
    bias_shape.push(output_features);
    let checkpoint_bias = bias.as_deref().map(|name| tensors.get(name)).transpose()?;
    if let Some(bias) = &checkpoint_bias {
        require(bias, Dtype::Bfloat16, &bias_shape, binding, "bias")?;
    }
    let weight = if weight_dtype == Dtype::Uint8 {
        let graph = stream.native().graph();
        let packed = graph.view_dtype(weight.native(), mirtal::DType::Uint32)?;
        let mut shape = bias_shape.clone();
        shape.push(input_features / 8);
        Array::from_native(graph.reshape(&packed, &mirtal::Shape::new(shape)?)?)?
    } else {
        weight
    };
    let has_bias = checkpoint_bias.is_some();
    let bias = checkpoint_bias.map_or_else(
        || {
            Array::from_native(stream.native().graph().full(
                &mirtal::Shape::new(bias_shape)?,
                0.0,
                mirtal::DType::Bfloat16,
            )?)
        },
        Ok,
    )?;
    Ok(MxFp4Linear {
        weight,
        scales,
        bias,
        input_features,
        output_features,
        has_bias,
        layout,
    })
}

pub(super) fn embedding(
    tensors: &ModelTensors,
    binding: &TensorBinding,
    stream: &Stream,
) -> Result<MxFp4Embedding> {
    let TensorStorage::BlockQuantized { bias: None, .. } = &binding.storage else {
        return Err(invalid(binding, "MXFP4 embedding cannot have output bias"));
    };
    Ok(MxFp4Embedding {
        linear: linear(tensors, binding, stream)?,
    })
}

impl MxFp4Embedding {
    pub(super) fn lookup(&self, indices: &Array, stream: &Stream) -> Result<Array> {
        stream.kernels().mxfp4_embedding(
            &self.linear.weight,
            &self.linear.scales,
            indices,
            self.linear.input_features,
            stream,
        )
    }

    pub(super) fn project(&self, input: &Array, stream: &Stream) -> Result<Array> {
        self.linear.forward(input, stream)
    }
}

fn projection_shape(
    binding: &TensorBinding,
) -> Result<(MxFp4LinearLayout, Vec<usize>, usize, usize)> {
    let (layout, prefix, output, input) =
        match (binding.block_projection_layout(), binding.logical_shape.as_deref()) {
            (Some(BlockProjectionLayout::Matrix), Some([output, input])) => {
                (MxFp4LinearLayout::Matrix, Vec::new(), *output, *input)
            },
            (
                Some(BlockProjectionLayout::MatrixBank { matrices }),
                Some([actual, output, input]),
            ) if matrices == *actual => {
                (MxFp4LinearLayout::Gathered, vec![matrices], *output, *input)
            },
            (
                Some(BlockProjectionLayout::FusedGateUpBank { experts, interleaved: true }),
                Some([actual, output, input]),
            ) if experts == *actual => {
                (MxFp4LinearLayout::Gathered, vec![experts], *output, *input)
            },
            _ => return Err(invalid(binding, "requires an ordinary or gathered matrix layout")),
        };
    if !input.is_multiple_of(BlockQuantization::MXFP4.block_size) {
        return Err(invalid(binding, "input width is not a complete MXFP4 block"));
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
