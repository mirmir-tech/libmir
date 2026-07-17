use super::{Kernels, template};
use crate::engine::{Error, Result};

impl Kernels {
    pub(crate) fn gated_delta_gates(
        &self,
        stream: &mirtal::Stream,
        inputs: [&mirtal::Array; 4],
    ) -> Result<[mirtal::Array; 2]> {
        let [alpha, beta, a_log, dt_bias] = inputs;
        let shape = alpha.shape()?;
        let dimensions = shape.dimensions().to_vec();
        if dimensions.len() != 3
            || beta.shape()? != shape
            || a_log.shape()? != dt_bias.shape()?
            || a_log.shape()?.dimensions() != [dimensions[2]]
        {
            return Err(Error::InvalidModel("Gated Delta gate shapes are incompatible".into()));
        }
        Ok(self.gated_delta_gates.dispatch(
            stream,
            inputs,
            &[
                mirtal::OutputSpec::new(shape.clone(), mirtal::DType::Float32),
                mirtal::OutputSpec::new(shape, mirtal::DType::Float32),
            ],
            &mirtal::Dispatch::new([alpha.len(), 1, 1], [256, 1, 1])
                .templates([template("HV", dimensions[2])?]),
        )?)
    }

    pub(crate) fn gated_delta_recurrence(
        &self,
        stream: &mirtal::Stream,
        inputs: [&mirtal::Array; 6],
    ) -> Result<[mirtal::Array; 2]> {
        let [query, key, value, .., state] = inputs;
        let key_shape = key.shape()?;
        let value_shape = value.shape()?;
        let key_dimensions = key_shape.dimensions().to_vec();
        let value_dimensions = value_shape.dimensions().to_vec();
        validate_recurrence(inputs, &key_dimensions, &value_dimensions)?;
        Ok(self.gated_delta_recurrence.dispatch(
            stream,
            inputs,
            &[
                mirtal::OutputSpec::new(value_shape, query.dtype()?),
                mirtal::OutputSpec::new(state.shape()?, state.dtype()?),
            ],
            &mirtal::Dispatch::new(
                [32, value_dimensions[3], value_dimensions[0] * value_dimensions[2]],
                [32, 4, 1],
            )
            .templates([
                mirtal::TemplateArg::dtype("InT", query.dtype()?),
                mirtal::TemplateArg::dtype("StT", state.dtype()?),
                template("DK", key_dimensions[3])?,
                template("DV", value_dimensions[3])?,
                template("HK", key_dimensions[2])?,
                template("HV", value_dimensions[2])?,
                template("STEPS", value_dimensions[1])?,
            ]),
        )?)
    }

    pub(crate) fn gated_delta_decode(
        &self,
        stream: &mirtal::Stream,
        inputs: [&mirtal::Array; 8],
        normalize: bool,
    ) -> Result<[mirtal::Array; 2]> {
        let [query, _, value, .., state] = inputs;
        let query_shape = query.shape()?;
        let value_shape = value.shape()?;
        let query_dimensions = query_shape.dimensions().to_vec();
        let value_dimensions = value_shape.dimensions().to_vec();
        validate_decode(inputs, &query_dimensions, &value_dimensions)?;
        Ok(self.gated_delta_decode.dispatch(
            stream,
            inputs,
            &[
                mirtal::OutputSpec::new(value_shape, query.dtype()?),
                mirtal::OutputSpec::new(state.shape()?, mirtal::DType::Float32),
            ],
            &mirtal::Dispatch::new(
                [256, 1, value_dimensions[0] * value_dimensions[2]],
                [256, 1, 1],
            )
            .templates([
                mirtal::TemplateArg::dtype("InT", query.dtype()?),
                mirtal::TemplateArg::dtype("StT", mirtal::DType::Float32),
                template("DK", query_dimensions[3])?,
                template("DV", value_dimensions[3])?,
                template("HK", query_dimensions[2])?,
                template("HV", value_dimensions[2])?,
                mirtal::TemplateArg::bool("NORMALIZE", normalize),
            ]),
        )?)
    }
}

fn validate_recurrence(inputs: [&mirtal::Array; 6], key: &[usize], value: &[usize]) -> Result<()> {
    let [query, keys, _, decay, update, state] = inputs;
    let ranks = key.len() == 4 && value.len() == 4;
    let expected_state = ranks.then(|| [value[0], value[2], value[3], key[3]]);
    let state_shape = state.shape()?;
    if query.shape()? != keys.shape()?
        || !ranks
        || key[0..2] != value[0..2]
        || !key[3].is_multiple_of(32)
        || decay.shape()? != update.shape()?
        || decay.shape()?.dimensions() != [value[0], value[1], value[2]]
        || expected_state.as_ref().is_none_or(|shape| state_shape.dimensions() != shape)
    {
        return Err(Error::InvalidModel("Gated Delta recurrence shapes are incompatible".into()));
    }
    Ok(())
}

fn validate_decode(inputs: [&mirtal::Array; 8], query: &[usize], value: &[usize]) -> Result<()> {
    let [queries, keys, _, alpha, beta, a_log, dt_bias, state] = inputs;
    let ranks = query.len() == 4 && value.len() == 4;
    let expected_state = ranks.then(|| [value[0], value[2], value[3], query[3]]);
    let state_shape = state.shape()?;
    if queries.shape()? != keys.shape()?
        || !ranks
        || query[1] != 1
        || value[1] != 1
        || !query[3].is_multiple_of(32)
        || alpha.shape()? != beta.shape()?
        || alpha.shape()?.dimensions() != [value[0], 1, value[2]]
        || a_log.shape()? != dt_bias.shape()?
        || a_log.shape()?.dimensions() != [value[2]]
        || state.dtype()? != mirtal::DType::Float32
        || expected_state.as_ref().is_none_or(|shape| state_shape.dimensions() != shape)
    {
        return Err(Error::InvalidModel("Gated Delta decode shapes are incompatible".into()));
    }
    Ok(())
}
