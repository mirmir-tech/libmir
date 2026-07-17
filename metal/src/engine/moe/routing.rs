use mirtal::{Array as NativeArray, DType, Error as NativeError, Graph};

use super::super::{Array, Result, Stream};

#[derive(Debug)]
pub struct RouterOutput {
    pub indices: Array,
    pub weights: Array,
}

impl Array {
    pub fn router_top_k(
        &self,
        expert_scale: &Self,
        top_k: i32,
        stream: &Stream,
    ) -> Result<RouterOutput> {
        let [indices, weights] = route_top_k(
            stream.native().graph(),
            self.native(),
            Some(expert_scale.native()),
            usize::try_from(top_k)?,
        )?;
        Ok(RouterOutput {
            indices: Self::from_native(indices)?,
            weights: Self::from_native(weights)?,
        })
    }

    pub fn router_top_k_unit(&self, top_k: i32, stream: &Stream) -> Result<RouterOutput> {
        let [indices, weights] =
            route_top_k(stream.native().graph(), self.native(), None, usize::try_from(top_k)?)?;
        Ok(RouterOutput {
            indices: Self::from_native(indices)?,
            weights: Self::from_native(weights)?,
        })
    }
}

pub(in crate::engine) fn route_top_k(
    graph: Graph<'_>,
    scores: &NativeArray,
    scale: Option<&NativeArray>,
    top_k: usize,
) -> mirtal::Result<[NativeArray; 2]> {
    let shape = scores.shape()?;
    let experts = shape
        .dimensions()
        .last()
        .copied()
        .ok_or_else(|| invalid("router scores must have rank"))?;
    if top_k == 0 || top_k > experts {
        return Err(invalid("router top-k is outside score dimensions"));
    }
    if let Some(scale) = scale
        && scale.shape()?.dimensions() != [experts]
    {
        return Err(invalid("router scale is incompatible with score dimensions"));
    }
    let kth = i32::try_from(experts - top_k)?;
    let indices = graph.argpartition(scores, kth, -1)?;
    let mut start = vec![0; shape.dimensions().len()];
    let mut stop = shape.dimensions().to_vec();
    let last = start.len() - 1;
    start[last] = experts - top_k;
    stop[last] = experts;
    let indices = graph.slice(&indices, &start, &stop)?;
    let selected = graph.take_along_axis(scores, &indices, -1)?;
    let mut weights = graph.softmax(&selected, -1, false)?;
    let indices = graph.astype(&indices, DType::Uint32)?;
    if let Some(scale) = scale {
        let scale = graph.astype(scale, scores.dtype()?)?;
        weights = graph.multiply(&graph.take(&scale, &indices, 0)?, &weights)?;
    }
    Ok([indices, weights])
}

fn invalid(message: impl Into<String>) -> NativeError {
    NativeError::InvalidOperation(message.into())
}
