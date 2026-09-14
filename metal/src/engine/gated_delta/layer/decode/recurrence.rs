use mirtal::{Array, DType, Dispatch, Graph, MetalKernel, OutputSpec, TemplateArg};

use crate::engine::{Result, kernels::gated_delta::new_gated_delta_decode_kernel};

pub(super) enum Recurrence {
    Native(MetalKernel<8, 2>),
    #[cfg(test)]
    Packed(MetalKernel<8, 2>),
}

impl Recurrence {
    pub(super) fn new() -> Result<Self> {
        Ok(Self::Native(new_gated_delta_decode_kernel()?))
    }

    #[cfg(test)]
    pub(super) fn packed() -> Result<Self> {
        Ok(Self::Packed(
            crate::engine::kernels::gated_delta::experiment::decode::packed_decode()?,
        ))
    }

    pub(super) fn dispatch(
        &self,
        graph: Graph<'_>,
        inputs: [&Array; 8],
    ) -> mirtal::Result<[Array; 2]> {
        let query = inputs[0];
        let value = inputs[2];
        let query_shape = query.shape()?;
        let value_shape = value.shape()?;
        let query_dimensions = query_shape.dimensions().to_vec();
        let value_dimensions = value_shape.dimensions().to_vec();
        #[cfg(test)]
        if let Self::Packed(kernel) = self {
            let dispatch =
                crate::engine::kernels::gated_delta::experiment::decode::dispatch(inputs, true)
                    .map_err(|error| mirtal::Error::InvalidDispatch(error.to_string()))?;
            return kernel.dispatch_graph(
                graph,
                inputs,
                &[
                    OutputSpec::new(value_shape, query.dtype()?),
                    OutputSpec::new(inputs[7].shape()?, DType::Float32),
                ],
                &dispatch,
            );
        }
        let kernel = match self {
            Self::Native(kernel) => kernel,
            #[cfg(test)]
            Self::Packed(_) => unreachable!("packed dispatch returned above"),
        };
        kernel.dispatch_graph(
            graph,
            inputs,
            &[
                OutputSpec::new(value_shape, query.dtype()?),
                OutputSpec::new(inputs[7].shape()?, DType::Float32),
            ],
            &Dispatch::new([256, 1, value_dimensions[0] * value_dimensions[2]], [256, 1, 1])
                .templates([
                    TemplateArg::dtype("InT", query.dtype()?),
                    TemplateArg::dtype("StT", DType::Float32),
                    TemplateArg::int("DK", i32::try_from(query_dimensions[3])?),
                    TemplateArg::int("DV", i32::try_from(value_dimensions[3])?),
                    TemplateArg::int("HK", i32::try_from(query_dimensions[2])?),
                    TemplateArg::int("HV", i32::try_from(value_dimensions[2])?),
                    TemplateArg::bool("NORMALIZE", true),
                ]),
        )
    }
}
