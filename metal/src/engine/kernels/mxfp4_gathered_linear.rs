use super::Kernels;
use crate::engine::{Array, Dtype, Error, Result, Stream};

mirtal::metal_kernel! {
    fn mxfp4_gathered_linear {
        name: "mirmir_mxfp4_gathered_linear",
        templates: [
            T: dtype = bf16, W: dtype = u8, INPUT: int = 2880, OUTPUT: int = 2880,
            MATRICES: int = 32, SELECTIONS: int = 1,
        ],
        inputs: [input: T, weight: W, scales: u8, bias: T, indices: u32],
        outputs: [output: T],
        source: file "kernels/mxfp4_gathered_linear.metal",
        header: inline "",
        row_contiguous: true,
        atomic_outputs: false,
    }
}

#[derive(Debug)]
pub(super) struct MxFp4GatheredLinearKernel {
    kernel: mirtal::MetalKernel<5, 1>,
}

impl MxFp4GatheredLinearKernel {
    pub(super) fn new() -> Result<Self> {
        Ok(Self { kernel: mxfp4_gathered_linear()? })
    }

    fn execute(
        &self,
        inputs: [&Array; 5],
        input_features: usize,
        output_features: usize,
        matrices: usize,
        stream: &Stream,
    ) -> Result<Array> {
        let input_shape = inputs[0].shape()?;
        let Some((&physical_input, matrix_shape)) = input_shape.split_last() else {
            return Err(Error::InvalidModel("gathered MXFP4 input shape is empty".into()));
        };
        if usize::try_from(physical_input)? != input_features {
            return Err(Error::InvalidModel("gathered MXFP4 input width differs from plan".into()));
        }
        let Some((&rows, input_prefix)) = matrix_shape.split_last() else {
            return Err(Error::InvalidModel("gathered MXFP4 input must have matrix rank".into()));
        };
        let indices_shape = inputs[4].shape()?;
        let input_rows = product(input_prefix)?;
        let assignments = product(&indices_shape)?;
        if inputs[4].dtype()? != Dtype::Uint32
            || rows != 1
            || input_prefix.len() != indices_shape.len()
            || !input_prefix
                .iter()
                .zip(&indices_shape)
                .all(|(input, selected)| *input == 1 || input == selected)
            || input_rows == 0
            || !assignments.is_multiple_of(input_rows)
        {
            return Err(Error::InvalidModel("gathered MXFP4 indices differ from input".into()));
        }
        let selections = assignments / input_rows;
        let mut output_shape = indices_shape
            .iter()
            .copied()
            .map(usize::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        output_shape.push(usize::try_from(rows)?);
        output_shape.push(output_features);
        let output =
            mirtal::OutputSpec::new(mirtal::Shape::new(output_shape)?, mirtal::DType::Bfloat16);
        let templates = [
            mirtal::TemplateArg::dtype("T", mirtal::DType::Bfloat16),
            mirtal::TemplateArg::dtype("W", inputs[1].native().dtype()?),
            integer("INPUT", input_features)?,
            integer("OUTPUT", output_features)?,
            integer("MATRICES", matrices)?,
            integer("SELECTIONS", selections)?,
        ];
        let [output] = self.kernel.dispatch(
            stream.native(),
            inputs.map(Array::native),
            &[output],
            &mirtal::Dispatch::new([output_features * 32, assignments, 1], [32, 1, 1])
                .templates(templates),
        )?;
        Array::from_native(output)
    }
}

impl Kernels {
    pub(crate) fn mxfp4_gathered_linear(
        &self,
        inputs: [&Array; 5],
        input_features: usize,
        output_features: usize,
        matrices: usize,
        stream: &Stream,
    ) -> Result<Array> {
        self.mxfp4_gathered_linear
            .execute(inputs, input_features, output_features, matrices, stream)
    }
}

fn product(shape: &[i32]) -> Result<usize> {
    shape.iter().try_fold(1_usize, |total, value| {
        total.checked_mul(usize::try_from(*value)?).ok_or(Error::ShapeOverflow)
    })
}

fn integer(name: &'static str, value: usize) -> Result<mirtal::TemplateArg> {
    Ok(mirtal::TemplateArg::int(name, i32::try_from(value)?))
}
