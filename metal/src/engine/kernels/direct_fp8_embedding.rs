use super::Kernels;
use crate::engine::{Array, Dtype, Error, Result, Stream};

mirtal::metal_kernel! {
    fn direct_fp8_embedding {
        name: "mirmir_direct_fp8_embedding",
        templates: [
            T: dtype = bf16, VOCAB: int = 32000, HIDDEN: int = 896,
            SCALE_STRIDE: int = 1,
            SCALE_GRID: int = 0, OUTPUT_BLOCK: int = 1,
            INPUT_BLOCK: int = 1, INPUT_GROUPS: int = 1, WEIGHT_E5M2: int = 1,
        ],
        inputs: [weight: u8, scales: f32, indices: u32],
        outputs: [output: T],
        source: file "kernels/direct_fp8_embedding.metal",
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
pub(super) struct DirectFp8EmbeddingKernel {
    kernel: mirtal::MetalKernel<3, 1>,
}

#[derive(Clone, Copy, Debug)]
pub(in crate::engine) struct DirectFp8EmbeddingSpec {
    pub vocab: usize,
    pub hidden: usize,
    pub scale_stride: usize,
    pub scale_grid: bool,
    pub output_block: usize,
    pub input_block: usize,
    pub input_groups: usize,
    pub weight_e5m2: bool,
}

impl DirectFp8EmbeddingKernel {
    pub(super) fn new() -> Result<Self> {
        Ok(Self { kernel: direct_fp8_embedding()? })
    }

    fn execute(
        &self,
        weight: &Array,
        scales: &Array,
        indices: &Array,
        spec: DirectFp8EmbeddingSpec,
        stream: &Stream,
    ) -> Result<Array> {
        if indices.dtype()? != Dtype::Uint32 {
            return Err(Error::InvalidModel("direct FP8 embedding indices must be U32".into()));
        }
        let shape = indices.shape()?;
        let tokens = shape.iter().try_fold(1_usize, |total, dimension| {
            total.checked_mul(usize::try_from(*dimension)?).ok_or(Error::ShapeOverflow)
        })?;
        let mut output_shape = shape
            .iter()
            .copied()
            .map(usize::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        output_shape.push(spec.hidden);
        let output =
            mirtal::OutputSpec::new(mirtal::Shape::new(output_shape)?, mirtal::DType::Bfloat16);
        let templates = [
            mirtal::TemplateArg::dtype("T", mirtal::DType::Bfloat16),
            integer("VOCAB", spec.vocab)?,
            integer("HIDDEN", spec.hidden)?,
            integer("SCALE_STRIDE", spec.scale_stride)?,
            integer("SCALE_GRID", usize::from(spec.scale_grid))?,
            integer("OUTPUT_BLOCK", spec.output_block)?,
            integer("INPUT_BLOCK", spec.input_block)?,
            integer("INPUT_GROUPS", spec.input_groups)?,
            integer("WEIGHT_E5M2", usize::from(spec.weight_e5m2))?,
        ];
        let [output] = self.kernel.dispatch(
            stream.native(),
            [weight.native(), scales.native(), indices.native()],
            &[output],
            &mirtal::Dispatch::new([spec.hidden, tokens, 1], [spec.hidden.min(256), 1, 1])
                .templates(templates),
        )?;
        Array::from_native(output)
    }
}

impl Kernels {
    pub(crate) fn direct_fp8_embedding(
        &self,
        weight: &Array,
        scales: &Array,
        indices: &Array,
        spec: DirectFp8EmbeddingSpec,
        stream: &Stream,
    ) -> Result<Array> {
        self.direct_fp8_embedding.execute(weight, scales, indices, spec, stream)
    }
}

fn integer(name: &'static str, value: usize) -> Result<mirtal::TemplateArg> {
    Ok(mirtal::TemplateArg::int(name, i32::try_from(value)?))
}
