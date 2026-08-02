use super::Kernels;
use crate::engine::{Array, Result, Stream};

mirtal::metal_kernel! {
    fn nvfp4_convert {
        name: "mirmir_nvfp4_convert",
        templates: [T: dtype = bf16],
        inputs: [weight: u8, scales: u8, global_scale: f32],
        outputs: [output: T],
        source: file "kernels/nvfp4_convert.metal",
        header: inline r"
            inline float mirmir_e4m3_to_float(uchar encoded) {
              uint magnitude = uint(encoded & 0x7fu);
              uint exponent = magnitude >> 3;
              uint mantissa = magnitude & 7u;
              float value = exponent == 0u
                  ? ldexp(float(mantissa), -9)
                  : ldexp(float(8u + mantissa), int(exponent) - 10);
              return (encoded & 0x80u) == 0u ? value : -value;
            }
        ",
        row_contiguous: true,
        atomic_outputs: false,
    }
}

#[derive(Debug)]
pub(super) struct NvFp4ConvertKernel {
    kernel: mirtal::MetalKernel<3, 1>,
}

impl NvFp4ConvertKernel {
    pub(super) fn new() -> Result<Self> {
        Ok(Self { kernel: nvfp4_convert()? })
    }

    fn execute(
        &self,
        weight: &Array,
        scales: &Array,
        global_scale: &Array,
        shape: [usize; 2],
        stream: &Stream,
    ) -> Result<Array> {
        let [output, input] = shape;
        let elements = output.checked_mul(input).ok_or(crate::engine::Error::ShapeOverflow)?;
        let output = mirtal::OutputSpec::new(mirtal::Shape::new(shape)?, mirtal::DType::Bfloat16);
        let [output] = self.kernel.dispatch(
            stream.native(),
            [weight.native(), scales.native(), global_scale.native()],
            &[output],
            &mirtal::Dispatch::new([elements, 1, 1], [256, 1, 1])
                .templates([mirtal::TemplateArg::dtype("T", mirtal::DType::Bfloat16)]),
        )?;
        Array::from_native(output)
    }
}

impl Kernels {
    pub(crate) fn nvfp4_convert(
        &self,
        weight: &Array,
        scales: &Array,
        global_scale: &Array,
        shape: [usize; 2],
        stream: &Stream,
    ) -> Result<Array> {
        self.nvfp4_convert.execute(weight, scales, global_scale, shape, stream)
    }
}
