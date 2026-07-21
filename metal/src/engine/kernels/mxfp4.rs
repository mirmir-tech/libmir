use super::{Kernels, template};
use crate::engine::{Array, Error, Result, Stream};

#[derive(Debug, Clone, Copy)]
pub struct MxFp4Shape {
    pub tokens: usize,
    pub top_k: usize,
    pub hidden: usize,
    pub intermediate: usize,
}

impl Kernels {
    pub(crate) fn mxfp4_gate_up(
        &self,
        inputs: [&Array; 6],
        shape: MxFp4Shape,
        stream: &Stream,
    ) -> Result<Array> {
        validate(shape)?;
        let output = mirtal::OutputSpec::new(
            mirtal::Shape::new([shape.tokens, shape.top_k, shape.intermediate])?,
            inputs[0].native().dtype()?,
        );
        let [output] = self.mxfp4_gate_up.dispatch(
            stream.native(),
            inputs.map(Array::native),
            &[output],
            &mirtal::Dispatch::new(
                [shape.intermediate * 32, shape.top_k, shape.tokens],
                [32, 1, 1],
            )
            .templates(templates(shape, inputs[0].native().dtype()?)?),
        )?;
        Array::from_native(output)
    }

    pub(crate) fn mxfp4_down(
        &self,
        inputs: [&Array; 6],
        shape: MxFp4Shape,
        stream: &Stream,
    ) -> Result<Array> {
        validate(shape)?;
        let output = mirtal::OutputSpec::new(
            mirtal::Shape::new([shape.tokens, shape.hidden])?,
            inputs[0].native().dtype()?,
        );
        let [output] = self.mxfp4_down.dispatch(
            stream.native(),
            inputs.map(Array::native),
            &[output],
            &mirtal::Dispatch::new([shape.hidden * 32, shape.tokens, 1], [32, 1, 1])
                .templates(templates(shape, inputs[0].native().dtype()?)?),
        )?;
        Array::from_native(output)
    }

    pub(crate) fn mxfp4_split_gate_up(
        &self,
        inputs: [&Array; 9],
        shape: MxFp4Shape,
        stream: &Stream,
    ) -> Result<Array> {
        validate(shape)?;
        let output = mirtal::OutputSpec::new(
            mirtal::Shape::new([shape.tokens, shape.top_k, shape.intermediate])?,
            inputs[0].native().dtype()?,
        );
        let [output] = self.mxfp4_split_gate_up.dispatch(
            stream.native(),
            inputs.map(Array::native),
            &[output],
            &mirtal::Dispatch::new(
                [shape.intermediate * 32, shape.top_k, shape.tokens],
                [32, 1, 1],
            )
            .templates(templates(shape, inputs[0].native().dtype()?)?),
        )?;
        Array::from_native(output)
    }

    pub(crate) fn mxfp4_u32_down(
        &self,
        inputs: [&Array; 6],
        shape: MxFp4Shape,
        stream: &Stream,
    ) -> Result<Array> {
        validate(shape)?;
        let output = mirtal::OutputSpec::new(
            mirtal::Shape::new([shape.tokens, shape.hidden])?,
            inputs[0].native().dtype()?,
        );
        let [output] = self.mxfp4_u32_down.dispatch(
            stream.native(),
            inputs.map(Array::native),
            &[output],
            &mirtal::Dispatch::new([shape.hidden * 32, shape.tokens, 1], [32, 1, 1])
                .templates(templates(shape, inputs[0].native().dtype()?)?),
        )?;
        Array::from_native(output)
    }
}

fn templates(shape: MxFp4Shape, dtype: mirtal::DType) -> Result<[mirtal::TemplateArg; 4]> {
    Ok([
        mirtal::TemplateArg::dtype("T", dtype),
        template("HIDDEN", shape.hidden)?,
        template("INTERMEDIATE", shape.intermediate)?,
        template("TOP_K", shape.top_k)?,
    ])
}

fn validate(shape: MxFp4Shape) -> Result<()> {
    if shape.tokens == 0
        || shape.top_k == 0
        || shape.hidden == 0
        || shape.intermediate == 0
        || !shape.hidden.is_multiple_of(32)
        || !shape.intermediate.is_multiple_of(32)
    {
        Err(Error::InvalidModel("invalid MXFP4 expert geometry".into()))
    } else {
        Ok(())
    }
}
