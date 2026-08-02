use models::weights::{BlockProjectionLayout, BlockQuantization, TensorBinding, TensorStorage};

use crate::engine::{Array, DenseLinear, Dtype, Error, ModelTensors, Result, Stream};

#[derive(Debug)]
pub(in crate::engine) struct NvFp4Linear {
    weight: Array,
    scales: Array,
    global_scale: Array,
    matrices: usize,
    input_features: usize,
    output_features: usize,
    per_matrix_global: bool,
}

struct Source {
    weight: Array,
    scales: Array,
    global_scale: Array,
}

pub(super) enum NvFp4Fallback {
    Dense(DenseLinear),
    Gathered(NvFp4Linear),
}

pub(super) fn linear(
    tensors: &ModelTensors,
    binding: &TensorBinding,
    stream: &Stream,
) -> Result<NvFp4Fallback> {
    let (matrices, output, input) = projection_shape(binding)?;
    let source = source(tensors, binding, matrices, output, input, stream)?;
    if let Some(matrices) = matrices {
        return Ok(NvFp4Fallback::Gathered(NvFp4Linear {
            weight: source.weight,
            scales: source.scales,
            global_scale: source.global_scale,
            matrices,
            input_features: input,
            output_features: output,
            per_matrix_global: false,
        }));
    }
    let weight = stream.kernels().nvfp4_convert(
        &source.weight,
        &source.scales,
        &source.global_scale,
        [output, input],
        stream,
    )?;
    weight.async_eval()?;
    DenseLinear::from_binding_weight(weight, None, false, stream).map(NvFp4Fallback::Dense)
}

pub(super) fn individual_bank(
    tensors: &ModelTensors,
    bindings: &[&TensorBinding],
    stream: &Stream,
) -> Result<NvFp4Linear> {
    let Some(first) = bindings.first() else {
        return Err(Error::InvalidQuantization("NVFP4 expert bank is empty".into()));
    };
    let (matrices, output, input) = projection_shape(first)?;
    if matrices.is_some() {
        return Err(invalid(first, "individual expert is not an ordinary matrix"));
    }
    let sources = bindings
        .iter()
        .map(|binding| {
            let (matrices, actual_output, actual_input) = projection_shape(binding)?;
            if matrices.is_some() || actual_output != output || actual_input != input {
                return Err(invalid(binding, "expert matrix geometry differs within bank"));
            }
            source(tensors, binding, None, output, input, stream)
        })
        .collect::<Result<Vec<_>>>()?;
    let matrices = sources.len();
    let weights = sources.iter().map(|source| &source.weight).collect::<Vec<_>>();
    let scales = sources.iter().map(|source| &source.scales).collect::<Vec<_>>();
    let globals = sources.iter().map(|source| &source.global_scale).collect::<Vec<_>>();
    let weight = Array::concatenate(&weights, 0, stream)?.reshape(
        &[i32::try_from(matrices)?, i32::try_from(output)?, i32::try_from(input / 2)?],
        stream,
    )?;
    let scales = Array::concatenate(&scales, 0, stream)?.reshape(
        &[i32::try_from(matrices)?, i32::try_from(output)?, i32::try_from(input / 16)?],
        stream,
    )?;
    let global_scale = Array::concatenate(&globals, 0, stream)?;
    for array in [&weight, &scales, &global_scale] {
        array.async_eval()?;
    }
    Ok(NvFp4Linear {
        weight,
        scales,
        global_scale,
        matrices,
        input_features: input,
        output_features: output,
        per_matrix_global: true,
    })
}

impl NvFp4Linear {
    pub(in crate::engine) fn gather(
        &self,
        input: &Array,
        indices: &Array,
        stream: &Stream,
    ) -> Result<Array> {
        if input.dtype()? != Dtype::Bfloat16 || indices.dtype()? != Dtype::Uint32 {
            return Err(Error::InvalidQuantization(
                "gathered NVFP4 requires BF16 input and U32 indices".into(),
            ));
        }
        stream.kernels().nvfp4_gathered_linear(
            [input, &self.weight, &self.scales, &self.global_scale, indices],
            self.input_features,
            self.output_features,
            self.matrices,
            self.per_matrix_global,
            stream,
        )
    }
}

fn source(
    tensors: &ModelTensors,
    binding: &TensorBinding,
    matrices: Option<usize>,
    output: usize,
    input: usize,
    stream: &Stream,
) -> Result<Source> {
    let TensorStorage::BlockQuantized {
        format: BlockQuantization::NVFP4,
        scales,
        global_scale: Some(global_scale),
        input_scale: Some(input_scale),
        bias: None,
        ..
    } = &binding.storage
    else {
        return Err(invalid(binding, "requires complete bias-free NVFP4 storage"));
    };
    if !input.is_multiple_of(BlockQuantization::NVFP4.block_size) {
        return Err(invalid(binding, "input width is not a complete NVFP4 block"));
    }
    let mut weight_shape = matrices.into_iter().collect::<Vec<_>>();
    weight_shape.extend([output, input / 2]);
    let mut scale_shape = matrices.into_iter().collect::<Vec<_>>();
    scale_shape.extend([output, input / 16]);
    let weight = tensors.get(&binding.source)?;
    let scales = tensors.get(scales)?;
    let global_scale = tensors.get(global_scale)?;
    let input_scale = tensors.get(input_scale)?;
    require(&weight, Dtype::Uint8, &weight_shape, binding, "weight")?;
    require(&scales, Dtype::Uint8, &scale_shape, binding, "scales")?;
    require(&global_scale, Dtype::Float32, &[], binding, "global scale")?;
    require(&input_scale, Dtype::Float32, &[], binding, "input scale")?;
    Ok(Source {
        weight,
        scales,
        global_scale: global_scale.reshape(&[1], stream)?,
    })
}

fn projection_shape(binding: &TensorBinding) -> Result<(Option<usize>, usize, usize)> {
    match (binding.block_projection_layout(), binding.logical_shape.as_deref()) {
        (Some(BlockProjectionLayout::Matrix), Some([output, input])) => Ok((None, *output, *input)),
        (Some(BlockProjectionLayout::MatrixBank { matrices }), Some([actual, output, input]))
            if matrices == *actual =>
        {
            Ok((Some(matrices), *output, *input))
        },
        (
            Some(BlockProjectionLayout::FusedGateUpBank { experts, interleaved: true }),
            Some([actual, output, input]),
        ) if experts == *actual => Ok((Some(experts), *output, *input)),
        _ => Err(invalid(binding, "requires an ordinary or gathered matrix layout")),
    }
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
