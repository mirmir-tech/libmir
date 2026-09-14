use super::{Array, DType, Error, Result, Shape, SortedExpertInputs, Stream, dimensions, elements};
use crate::engine::kernels::expert_group::aligned::AlignedGroup;

impl Array {
    pub(crate) fn align_expert_inputs(
        &self,
        indices: &Self,
        experts: usize,
        plan: &AlignedGroup,
        stream: &Stream,
    ) -> Result<SortedExpertInputs> {
        let input_shape = dimensions(&self.shape()?)?;
        let routing_shape = dimensions(&indices.shape()?)?;
        if input_shape.len() != 3
            || routing_shape.len() != 3
            || input_shape[..2] != routing_shape[..2]
            || routing_shape[2] == 0
        {
            return Err(Error::InvalidModel(
                "aligned expert input and routing shapes do not align".into(),
            ));
        }
        let routes = elements(&routing_shape)?;
        let graph = stream.native().graph();
        let flat = graph.reshape(indices.native(), &Shape::new([routes])?)?;
        let [order, inverse, grouped_indices] = plan.forward(stream.native(), &flat, experts)?;
        let divisor = graph.full(
            &Shape::new([])?,
            f32::from(u16::try_from(routing_shape[2])?),
            DType::Uint32,
        )?;
        let rows = graph.floor_divide(&order, &divisor)?;
        let tokens = input_shape[0].checked_mul(input_shape[1]).ok_or(Error::ShapeOverflow)?;
        let source = graph.reshape(self.native(), &Shape::new([tokens, 1, input_shape[2]])?)?;
        Ok(SortedExpertInputs {
            input: Self::from_native(graph.take(&source, &rows, 0)?)?,
            indices: Self::from_native(grouped_indices)?,
            source: Self::from_native(source)?,
            rows: Self::from_native(rows)?,
            inverse: Self::from_native(inverse)?,
            routing_shape,
        })
    }
}
