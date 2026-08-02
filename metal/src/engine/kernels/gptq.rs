use super::Kernels;
use crate::engine::{Array, Error, QuantizedArrays, Result, Stream};

mirtal::metal_kernel! {
    fn gptq_repack {
        name: "mirmir_gptq_repack",
        templates: [INPUT: int = 1024, OUTPUT: int = 2048, GROUP: int = 128, LEGACY: int = 1],
        inputs: [qweight: u32, qzeros: u32, scales: f16],
        outputs: [weight: u32, native_scales: f16, biases: f16],
        source: file "kernels/gptq_repack.metal",
        header: inline "",
        row_contiguous: true,
        atomic_outputs: false,
    }
}

mirtal::metal_kernel! {
    fn gptq_linear {
        name: "mirmir_gptq_linear",
        templates: [
            T: dtype = bf16, INPUT: int = 1024, OUTPUT: int = 2048,
            GROUP: int = 128, LEGACY: int = 1,
        ],
        inputs: [input: T, qweight: u32, qzeros: u32, scales: f16, group_indices: i32],
        outputs: [output: T],
        source: file "kernels/gptq_linear.metal",
        header: inline "",
        row_contiguous: true,
        atomic_outputs: false,
    }
}

#[derive(Debug)]
pub(super) struct GptqKernels {
    repack: mirtal::MetalKernel<3, 3>,
    linear: mirtal::MetalKernel<5, 1>,
}

impl GptqKernels {
    pub(super) fn new() -> Result<Self> {
        Ok(Self {
            repack: gptq_repack()?,
            linear: gptq_linear()?,
        })
    }

    fn repack(
        &self,
        stream: &Stream,
        [qweight, qzeros, scales]: [&Array; 3],
        input: usize,
        output: usize,
        group: usize,
        legacy: bool,
    ) -> Result<QuantizedArrays> {
        validate(input, output, group)?;
        let words = input / 8;
        let groups = input / group;
        let outputs = [
            output_spec([output, words], mirtal::DType::Uint32)?,
            output_spec([output, groups], mirtal::DType::Float16)?,
            output_spec([output, groups], mirtal::DType::Float16)?,
        ];
        let [weight, scales, biases] = self.repack.dispatch(
            stream.native(),
            [qweight.native(), qzeros.native(), scales.native()],
            &outputs,
            &mirtal::Dispatch::new([words.max(groups), output, 1], [32.min(words), 1, 1])
                .templates([
                    super::template("INPUT", input)?,
                    super::template("OUTPUT", output)?,
                    super::template("GROUP", group)?,
                    super::template("LEGACY", usize::from(legacy))?,
                ]),
        )?;
        QuantizedArrays::new(
            Array::from_native(weight)?,
            Array::from_native(scales)?,
            Array::from_native(biases)?,
            i32::try_from(group)?,
            4,
        )
    }

    fn linear(
        &self,
        stream: &Stream,
        inputs: [&Array; 5],
        input: usize,
        output: usize,
        group: usize,
        legacy: bool,
    ) -> Result<Array> {
        validate(input, output, group)?;
        let mut shape = inputs[0].shape()?;
        let Some(width) = shape.last_mut() else {
            return Err(Error::InvalidQuantization("GPTQ input shape is empty".into()));
        };
        if usize::try_from(*width)? != input {
            return Err(Error::InvalidQuantization("GPTQ input width differs".into()));
        }
        *width = i32::try_from(output)?;
        let tokens = shape[..shape.len() - 1].iter().try_fold(1_usize, |total, value| {
            total.checked_mul(usize::try_from(*value)?).ok_or(Error::ShapeOverflow)
        })?;
        let specification = mirtal::OutputSpec::new(
            mirtal::Shape::new(
                shape
                    .into_iter()
                    .map(usize::try_from)
                    .collect::<std::result::Result<Vec<_>, _>>()?,
            )?,
            inputs[0].native().dtype()?,
        );
        let [result] = self.linear.dispatch(
            stream.native(),
            inputs.map(Array::native),
            &[specification],
            &mirtal::Dispatch::new([output * 32, tokens, 1], [32, 1, 1]).templates([
                mirtal::TemplateArg::dtype("T", inputs[0].native().dtype()?),
                super::template("INPUT", input)?,
                super::template("OUTPUT", output)?,
                super::template("GROUP", group)?,
                super::template("LEGACY", usize::from(legacy))?,
            ]),
        )?;
        Array::from_native(result)
    }
}

impl Kernels {
    pub(crate) fn gptq_repack(
        &self,
        stream: &Stream,
        inputs: [&Array; 3],
        input: usize,
        output: usize,
        group: usize,
        legacy: bool,
    ) -> Result<QuantizedArrays> {
        self.gptq.repack(stream, inputs, input, output, group, legacy)
    }

    pub(crate) fn gptq_linear(
        &self,
        stream: &Stream,
        inputs: [&Array; 5],
        input: usize,
        output: usize,
        group: usize,
        legacy: bool,
    ) -> Result<Array> {
        self.gptq.linear(stream, inputs, input, output, group, legacy)
    }
}

fn output_spec(shape: [usize; 2], dtype: mirtal::DType) -> Result<mirtal::OutputSpec> {
    Ok(mirtal::OutputSpec::new(mirtal::Shape::new(shape)?, dtype))
}

fn validate(input: usize, output: usize, group: usize) -> Result<()> {
    if input == 0
        || output == 0
        || group == 0
        || !input.is_multiple_of(group)
        || !input.is_multiple_of(8)
        || !output.is_multiple_of(8)
    {
        Err(Error::InvalidQuantization("invalid GPTQ repack geometry".into()))
    } else {
        Ok(())
    }
}
