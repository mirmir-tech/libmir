use super::Kernels;
use crate::engine::{Array, Error, Result, Stream};

mirtal::metal_kernel! {
    fn direct_fp8 {
        name: "mirmir_direct_fp8",
        templates: [
            T: dtype = bf16, A: dtype = u8, INPUT_FEATURES: int = 896,
            OUTPUT_FEATURES: int = 896, SCALE_STRIDE: int = 1,
            SCALE_GRID: int = 0, OUTPUT_BLOCK: int = 1,
            INPUT_BLOCK: int = 1, INPUT_GROUPS: int = 1,
            ACTIVATION_STRIDE: int = 1,
            ACTIVATION_FP8: int = 1, WEIGHT_E5M2: int = 0,
        ],
        inputs: [input: A, input_scales: f32, weight: u8, scales: f32, bias: bf16],
        outputs: [output: T],
        source: file "kernels/direct_fp8.metal",
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

            inline float mirmir_e5m2_to_float(uchar encoded) {
              uint magnitude = uint(encoded & 0x7fu);
              uint exponent = magnitude >> 2;
              uint mantissa = magnitude & 3u;
              float value;
              if (exponent == 0u) {
                value = ldexp(float(mantissa), -16);
              } else if (exponent == 31u) {
                value = mantissa == 0u ? INFINITY : NAN;
              } else {
                value = ldexp(float(4u + mantissa), int(exponent) - 17);
              }
              return (encoded & 0x80u) == 0u ? value : -value;
            }
        ",
        row_contiguous: true,
        atomic_outputs: false,
    }
}

#[derive(Debug)]
pub(super) struct DirectFp8Kernel {
    kernel: mirtal::MetalKernel<5, 1>,
}

#[derive(Debug, Clone, Copy)]
pub struct DirectFp8Spec {
    pub input_features: usize,
    pub output_features: usize,
    pub scale_stride: usize,
    pub scale_grid: bool,
    pub output_block: usize,
    pub input_block: usize,
    pub input_groups: usize,
    pub activation_stride: usize,
    pub activation_fp8: bool,
    pub weight_e5m2: bool,
}

impl DirectFp8Kernel {
    pub(super) fn new() -> Result<Self> {
        Ok(Self { kernel: direct_fp8()? })
    }

    fn execute(&self, inputs: [&Array; 5], spec: DirectFp8Spec, stream: &Stream) -> Result<Array> {
        let input_shape = inputs[0].shape()?;
        let Some((&physical_input, prefix)) = input_shape.split_last() else {
            return Err(Error::InvalidModel("direct FP8 input shape is empty".into()));
        };
        if usize::try_from(physical_input)? != spec.input_features {
            return Err(Error::InvalidModel("direct FP8 input width differs from plan".into()));
        }
        let tokens = prefix.iter().try_fold(1_usize, |total, dimension| {
            total.checked_mul(usize::try_from(*dimension)?).ok_or(Error::ShapeOverflow)
        })?;
        let mut output_shape = prefix
            .iter()
            .copied()
            .map(usize::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        output_shape.push(spec.output_features);
        let output =
            mirtal::OutputSpec::new(mirtal::Shape::new(output_shape)?, mirtal::DType::Bfloat16);
        let templates = [
            mirtal::TemplateArg::dtype("T", mirtal::DType::Bfloat16),
            mirtal::TemplateArg::dtype(
                "A",
                if spec.activation_fp8 {
                    mirtal::DType::Uint8
                } else {
                    mirtal::DType::Bfloat16
                },
            ),
            integer("INPUT_FEATURES", spec.input_features)?,
            integer("OUTPUT_FEATURES", spec.output_features)?,
            integer("SCALE_STRIDE", spec.scale_stride)?,
            integer("SCALE_GRID", usize::from(spec.scale_grid))?,
            integer("OUTPUT_BLOCK", spec.output_block)?,
            integer("INPUT_BLOCK", spec.input_block)?,
            integer("INPUT_GROUPS", spec.input_groups)?,
            integer("ACTIVATION_STRIDE", spec.activation_stride)?,
            integer("ACTIVATION_FP8", usize::from(spec.activation_fp8))?,
            integer("WEIGHT_E5M2", usize::from(spec.weight_e5m2))?,
        ];
        let [output] = self.kernel.dispatch(
            stream.native(),
            inputs.map(Array::native),
            &[output],
            &mirtal::Dispatch::new([spec.output_features * 32, tokens, 1], [32, 1, 1])
                .templates(templates),
        )?;
        Array::from_native(output)
    }
}

impl Kernels {
    pub(crate) fn direct_fp8(
        &self,
        inputs: [&Array; 5],
        spec: DirectFp8Spec,
        stream: &Stream,
    ) -> Result<Array> {
        self.direct_fp8.execute(inputs, spec, stream)
    }
}

fn integer(name: &'static str, value: usize) -> Result<mirtal::TemplateArg> {
    Ok(mirtal::TemplateArg::int(name, i32::try_from(value)?))
}
