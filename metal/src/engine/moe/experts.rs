use mirtal::{DType, Shape};

use super::super::{Array, Error, Result, Stream};

#[derive(Debug)]
pub struct SortedExpertInputs {
    pub input: Array,
    pub indices: Array,
    inverse: Array,
    routing_shape: Vec<usize>,
}

impl SortedExpertInputs {
    pub fn restore(&self, output: &Array, stream: &Stream) -> Result<Array> {
        let graph = stream.native().graph();
        let restored = graph.take(output.native(), self.inverse.native(), 0)?;
        let restored = graph.squeeze_axis(&restored, 1)?;
        let hidden = output.shape()?.last().copied().ok_or(Error::ShapeOverflow)?;
        let mut shape = self.routing_shape.clone();
        shape.push(usize::try_from(hidden)?);
        Array::from_native(graph.reshape(&restored, &Shape::new(shape)?)?)
    }

    pub fn restore_weighted(
        &self,
        output: &Array,
        weights: &Array,
        stream: &Stream,
    ) -> Result<Array> {
        Array::from_native(stream.expert_restore_reduce([
            output.native(),
            self.inverse.native(),
            weights.native(),
        ])?)
    }
}

impl Array {
    pub fn sort_expert_inputs(
        &self,
        indices: &Self,
        stream: &Stream,
    ) -> Result<SortedExpertInputs> {
        let input_shape = dimensions(&self.shape()?)?;
        let routing_shape = dimensions(&indices.shape()?)?;
        if input_shape.len() != 3
            || routing_shape.len() != 3
            || input_shape[..2] != routing_shape[..2]
        {
            return Err(Error::InvalidModel("expert input and routing shapes do not align".into()));
        }
        let routes = elements(&routing_shape)?;
        let graph = stream.native().graph();
        let flat_indices = graph.reshape(indices.native(), &Shape::new([routes])?)?;
        let order = graph.argsort(&flat_indices, 0)?;
        grouped(self, &flat_indices, &order, graph.argsort(&order, 0)?, routing_shape, stream)
    }

    pub fn group_expert_inputs(
        &self,
        indices: &Self,
        experts: usize,
        stream: &Stream,
    ) -> Result<SortedExpertInputs> {
        let input_shape = dimensions(&self.shape()?)?;
        let routing_shape = dimensions(&indices.shape()?)?;
        if input_shape.len() != 3
            || routing_shape.len() != 3
            || input_shape[..2] != routing_shape[..2]
        {
            return Err(Error::InvalidModel("expert input and routing shapes do not align".into()));
        }
        let routes = elements(&routing_shape)?;
        let graph = stream.native().graph();
        let flat = graph.reshape(indices.native(), &Shape::new([routes])?)?;
        let [order, inverse] = stream.expert_group(&flat, experts)?;
        grouped(self, &flat, &order, inverse, routing_shape, stream)
    }
}

fn grouped(
    input: &Array,
    flat_indices: &mirtal::Array,
    order: &mirtal::Array,
    inverse: mirtal::Array,
    routing_shape: Vec<usize>,
    stream: &Stream,
) -> Result<SortedExpertInputs> {
    let graph = stream.native().graph();
    let input_shape = dimensions(&input.shape()?)?;
    let route_width = f32::from(u16::try_from(routing_shape[2])?);
    let divisor = graph.full(&Shape::new([])?, route_width, DType::Uint32)?;
    let rows = graph.floor_divide(order, &divisor)?;
    let tokens = input_shape[0].checked_mul(input_shape[1]).ok_or(Error::ShapeOverflow)?;
    let flat_input = graph.reshape(input.native(), &Shape::new([tokens, 1, input_shape[2]])?)?;
    Ok(SortedExpertInputs {
        input: Array::from_native(graph.take(&flat_input, &rows, 0)?)?,
        indices: Array::from_native(graph.take(flat_indices, order, 0)?)?,
        inverse: Array::from_native(inverse)?,
        routing_shape,
    })
}

impl Array {
    pub fn weighted_sum(&self, weights: &Self, axis: i32, stream: &Stream) -> Result<Self> {
        let graph = stream.native().graph();
        let weights = graph.expand_dims(weights.native(), &[-1])?;
        let weighted = graph.multiply(&weights, self.native())?;
        Self::from_native(graph.reduce_sum(&weighted, axis, false)?)?.astype_like(self, stream)
    }
}

fn dimensions(shape: &[i32]) -> Result<Vec<usize>> {
    Ok(shape
        .iter()
        .copied()
        .map(usize::try_from)
        .collect::<std::result::Result<_, _>>()?)
}

fn elements(shape: &[usize]) -> Result<usize> {
    shape
        .iter()
        .try_fold(1_usize, |total, value| total.checked_mul(*value).ok_or(Error::ShapeOverflow))
}
