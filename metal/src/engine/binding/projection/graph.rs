use mirtal::{Array, Graph};

use super::{BoundLinear, MxFp4LinearLayout};
use crate::engine::Result;

pub(in crate::engine) enum GraphLinear {
    Affine {
        quantized: mirtal::QuantizedArrays,
        bias: Option<Array>,
    },
    MxFp4 {
        quantized: mirtal::MxFp4Arrays,
        bias: Option<Array>,
    },
}

impl GraphLinear {
    pub(in crate::engine) fn new(linear: &BoundLinear) -> Result<Option<Self>> {
        match linear {
            BoundLinear::Affine(linear) => {
                let (quantized, bias) = linear.graph_parts()?;
                Ok(Some(Self::Affine { quantized, bias }))
            },
            BoundLinear::MxFp4(linear)
                if matches!(linear.layout, MxFp4LinearLayout::Matrix)
                    && linear.weight.native().dtype()? == mirtal::DType::Uint32 =>
            {
                let quantized = mirtal::MxFp4Arrays {
                    weight: linear.weight.native().clone(),
                    scales: linear.scales.native().clone(),
                };
                let bias = linear.has_bias.then(|| linear.bias.native().clone());
                Ok(Some(Self::MxFp4 { quantized, bias }))
            },
            _ => Ok(None),
        }
    }

    pub(in crate::engine) fn forward(
        &self,
        graph: Graph<'_>,
        input: &Array,
    ) -> mirtal::Result<Array> {
        let (output, bias) = match self {
            Self::Affine { quantized, bias } => {
                (graph.quantized_matmul(input, quantized.as_ref(), true)?, bias)
            },
            Self::MxFp4 { quantized, bias } => {
                (graph.mxfp4_matmul(input, quantized.as_ref(), true)?, bias)
            },
        };
        let output = graph.astype(&output, input.dtype()?)?;
        bias.as_ref().map_or(Ok(output.clone()), |bias| graph.add(&output, bias))
    }
}
