use super::Kernels;
use crate::engine::{Array, Error, Result, Stream};

mirtal::metal_kernel! {
    fn mxfp4_embedding {
        name: "mirmir_mxfp4_embedding",
        templates: [W: dtype = u8, HIDDEN: int = 2880],
        inputs: [weight: W, scales: u8, indices: u32],
        outputs: [output: bf16],
        source: file "kernels/mxfp4_embedding.metal",
        header: inline "",
        row_contiguous: true,
        atomic_outputs: false,
    }
}

#[derive(Debug)]
pub(super) struct MxFp4EmbeddingKernel {
    kernel: mirtal::MetalKernel<3, 1>,
}

impl MxFp4EmbeddingKernel {
    pub(super) fn new() -> Result<Self> {
        Ok(Self { kernel: mxfp4_embedding()? })
    }

    fn execute(
        &self,
        weight: &Array,
        scales: &Array,
        indices: &Array,
        hidden: usize,
        stream: &Stream,
    ) -> Result<Array> {
        if indices.dtype()? != crate::engine::Dtype::Uint32 {
            return Err(Error::InvalidModel("MXFP4 embedding indices must be U32".into()));
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
        output_shape.push(hidden);
        let output =
            mirtal::OutputSpec::new(mirtal::Shape::new(output_shape)?, mirtal::DType::Bfloat16);
        let [output] = self.kernel.dispatch(
            stream.native(),
            [weight.native(), scales.native(), indices.native()],
            &[output],
            &mirtal::Dispatch::new([hidden, tokens, 1], [hidden.min(256), 1, 1]).templates([
                mirtal::TemplateArg::dtype("W", weight.native().dtype()?),
                mirtal::TemplateArg::int("HIDDEN", i32::try_from(hidden)?),
            ]),
        )?;
        Array::from_native(output)
    }
}

impl Kernels {
    pub(crate) fn mxfp4_embedding(
        &self,
        weight: &Array,
        scales: &Array,
        indices: &Array,
        hidden: usize,
        stream: &Stream,
    ) -> Result<Array> {
        self.mxfp4_embedding.execute(weight, scales, indices, hidden, stream)
    }
}
