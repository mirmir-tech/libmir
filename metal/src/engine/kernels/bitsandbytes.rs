use super::Kernels;
use crate::engine::{Array, Error, Result, Stream};

mirtal::metal_kernel! {
    fn bitsandbytes_4bit_linear {
        name: "mirmir_bitsandbytes_4bit_linear",
        templates: [T: dtype = bf16, W: dtype = u8, A: dtype = f32,
            INPUT: int = 1024, OUTPUT: int = 1024, BLOCK: int = 64,
            NESTED: int = 0, OFFSET_BITS: int = 0],
        inputs: [input: T, weight: W, absmax: A, quant_map: f32,
            nested_absmax: f32, nested_quant_map: f32],
        outputs: [output: T],
        source: file "kernels/bitsandbytes_4bit_linear.metal",
        header: inline "",
        row_contiguous: true,
        atomic_outputs: false,
    }
}

#[derive(Debug)]
pub(super) struct BitsAndBytes4BitKernel {
    kernel: mirtal::MetalKernel<6, 1>,
}

impl BitsAndBytes4BitKernel {
    pub(super) fn new() -> Result<Self> {
        Ok(Self { kernel: bitsandbytes_4bit_linear()? })
    }

    #[allow(clippy::too_many_arguments)]
    fn execute(
        &self,
        inputs: [&Array; 6],
        input_features: usize,
        output_features: usize,
        block_size: usize,
        nested_block_size: Option<usize>,
        nested_offset_bits: u32,
        stream: &Stream,
    ) -> Result<Array> {
        let shape = inputs[0].shape()?;
        let Some((&physical_input, prefix)) = shape.split_last() else {
            return Err(Error::InvalidModel("bitsandbytes input shape is empty".into()));
        };
        if usize::try_from(physical_input)? != input_features {
            return Err(Error::InvalidModel("bitsandbytes input width differs".into()));
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
        let output_dtype = inputs[0].native().dtype()?;
        let output = mirtal::OutputSpec::new(mirtal::Shape::new(output_shape)?, output_dtype);
        let templates = [
            mirtal::TemplateArg::dtype("T", output_dtype),
            mirtal::TemplateArg::dtype("W", inputs[1].native().dtype()?),
            mirtal::TemplateArg::dtype("A", inputs[2].native().dtype()?),
            integer("INPUT", input_features)?,
            integer("OUTPUT", output_features)?,
            integer("BLOCK", block_size)?,
            integer("NESTED", nested_block_size.unwrap_or(0))?,
            mirtal::TemplateArg::int(
                "OFFSET_BITS",
                i32::from_ne_bytes(nested_offset_bits.to_ne_bytes()),
            ),
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
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn bitsandbytes_4bit_linear(
        &self,
        inputs: [&Array; 6],
        input_features: usize,
        output_features: usize,
        block_size: usize,
        nested_block_size: Option<usize>,
        nested_offset_bits: u32,
        stream: &Stream,
    ) -> Result<Array> {
        self.bitsandbytes_4bit.execute(
            inputs,
            input_features,
            output_features,
            block_size,
            nested_block_size,
            nested_offset_bits,
            stream,
        )
    }
}

fn integer(name: &'static str, value: usize) -> Result<mirtal::TemplateArg> {
    Ok(mirtal::TemplateArg::int(name, i32::try_from(value)?))
}
