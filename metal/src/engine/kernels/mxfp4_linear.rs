use super::Kernels;
use crate::engine::{Array, Error, Result, Stream};

mirtal::metal_kernel! {
    fn mxfp4_linear {
        name: "mirmir_mxfp4_linear",
        templates: [T: dtype = bf16, W: dtype = u8, INPUT: int = 2880, OUTPUT: int = 2880],
        inputs: [input: T, weight: W, scales: u8, bias: T],
        outputs: [output: T],
        source: file "kernels/mxfp4_linear.metal",
        header: inline "",
        row_contiguous: true,
        atomic_outputs: false,
    }
}

#[derive(Debug)]
pub(super) struct MxFp4LinearKernel {
    kernel: mirtal::MetalKernel<4, 1>,
}

impl MxFp4LinearKernel {
    pub(super) fn new() -> Result<Self> {
        Ok(Self { kernel: mxfp4_linear()? })
    }

    fn execute(
        &self,
        inputs: [&Array; 4],
        input_features: usize,
        output_features: usize,
        stream: &Stream,
    ) -> Result<Array> {
        let shape = inputs[0].shape()?;
        let Some((&physical_input, prefix)) = shape.split_last() else {
            return Err(Error::InvalidModel("MXFP4 input shape is empty".into()));
        };
        if usize::try_from(physical_input)? != input_features {
            return Err(Error::InvalidModel("MXFP4 input width differs from plan".into()));
        }
        let tokens = prefix.iter().try_fold(1_usize, |total, dimension| {
            total.checked_mul(usize::try_from(*dimension)?).ok_or(Error::ShapeOverflow)
        })?;
        let mut output_shape = prefix
            .iter()
            .copied()
            .map(usize::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        output_shape.push(output_features);
        let output =
            mirtal::OutputSpec::new(mirtal::Shape::new(output_shape)?, mirtal::DType::Bfloat16);
        let templates = [
            mirtal::TemplateArg::dtype("T", mirtal::DType::Bfloat16),
            mirtal::TemplateArg::dtype("W", inputs[1].native().dtype()?),
            integer("INPUT", input_features)?,
            integer("OUTPUT", output_features)?,
        ];
        let [output] = self.kernel.dispatch(
            stream.native(),
            inputs.map(Array::native),
            &[output],
            &mirtal::Dispatch::new([output_features * 32, tokens, 1], [32, 1, 1])
                .templates(templates),
        )?;
        Array::from_native(output)
    }
}

impl Kernels {
    pub(crate) fn mxfp4_linear(
        &self,
        inputs: [&Array; 4],
        input_features: usize,
        output_features: usize,
        stream: &Stream,
    ) -> Result<Array> {
        self.mxfp4_linear.execute(inputs, input_features, output_features, stream)
    }
}

fn integer(name: &'static str, value: usize) -> Result<mirtal::TemplateArg> {
    Ok(mirtal::TemplateArg::int(name, i32::try_from(value)?))
}
