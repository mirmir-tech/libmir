use super::Kernels;
use crate::engine::{Array, Error, QuantizedArrays, Result, Stream};

mirtal::metal_kernel! {
    fn awq_repack {
        name: "mirmir_awq_repack",
        templates: [INPUT: int = 1024, OUTPUT: int = 2048, GROUP: int = 128],
        inputs: [qweight: u32, qzeros: u32, scales: f16],
        outputs: [weight: u32, native_scales: f16, biases: f16],
        source: file "kernels/awq_repack.metal",
        header: inline "",
        row_contiguous: true,
        atomic_outputs: false,
    }
}

#[derive(Debug)]
pub(super) struct AwqRepackKernel {
    kernel: mirtal::MetalKernel<3, 3>,
}

impl AwqRepackKernel {
    pub(super) fn new() -> Result<Self> {
        Ok(Self { kernel: awq_repack()? })
    }

    fn repack(
        &self,
        stream: &Stream,
        [qweight, qzeros, scales]: [&Array; 3],
        input: usize,
        output: usize,
        group: usize,
    ) -> Result<QuantizedArrays> {
        validate(input, output, group)?;
        let words = input / 8;
        let groups = input / group;
        let outputs = [
            output_spec([output, words], mirtal::DType::Uint32)?,
            output_spec([output, groups], mirtal::DType::Float16)?,
            output_spec([output, groups], mirtal::DType::Float16)?,
        ];
        let [weight, scales, biases] = self.kernel.dispatch(
            stream.native(),
            [qweight.native(), qzeros.native(), scales.native()],
            &outputs,
            &mirtal::Dispatch::new([words.max(groups), output, 1], [32.min(words), 1, 1])
                .templates([
                    super::template("INPUT", input)?,
                    super::template("OUTPUT", output)?,
                    super::template("GROUP", group)?,
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
}

impl Kernels {
    pub(crate) fn awq_repack(
        &self,
        stream: &Stream,
        inputs: [&Array; 3],
        input: usize,
        output: usize,
        group: usize,
    ) -> Result<QuantizedArrays> {
        self.awq_repack.repack(stream, inputs, input, output, group)
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
        Err(Error::InvalidQuantization("invalid AWQ repack geometry".into()))
    } else {
        Ok(())
    }
}
