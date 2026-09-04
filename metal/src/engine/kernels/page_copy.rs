use super::Kernels;
use crate::engine::{Error, Result};

impl Kernels {
    pub(crate) fn copy_kv_page(
        &self,
        stream: &mirtal::Stream,
        inputs: [&mirtal::Array; 2],
        source: usize,
        target: usize,
    ) -> Result<[mirtal::Array; 2]> {
        let shape = inputs[0].shape()?;
        let dimensions = shape.dimensions();
        if !(3..=4).contains(&dimensions.len())
            || shape != inputs[1].shape()?
            || inputs[0].dtype()? != inputs[1].dtype()?
            || source == target
            || source >= dimensions[1]
            || target >= dimensions[1]
        {
            return Err(Error::InvalidModel("incompatible page-copy buffers or indices".into()));
        }
        let page_elements = dimensions[2..].iter().try_fold(1_usize, |size, dimension| {
            size.checked_mul(*dimension).ok_or(Error::ShapeOverflow)
        })?;
        let elements = dimensions[0].checked_mul(page_elements).ok_or(Error::ShapeOverflow)?;
        // The checked arenas are contiguous and use uint indexing in Metal.
        let extent = elements.checked_mul(dimensions[1]).ok_or(Error::ShapeOverflow)?;
        let _extent = u32::try_from(extent)?;
        let function = match inputs[0].dtype()? {
            mirtal::DType::Float16 | mirtal::DType::Bfloat16 => "mirmir_page_copy_16",
            mirtal::DType::Float32 | mirtal::DType::Uint32 => "mirmir_page_copy_32",
            _ => return Err(Error::InvalidModel("unsupported page-copy dtype".into())),
        };
        let dispatch = mirtal::AliasingDispatch::new([0, 1])
            .constants([
                u32::try_from(source)?,
                u32::try_from(target)?,
                u32::try_from(dimensions[1])?,
                u32::try_from(page_elements)?,
            ])
            .grid([elements, 1, 1])
            .threadgroup([elements.min(256), 1, 1]);
        Ok(self
            .paged_kv
            .export(function)?
            .dispatch_aliasing_array(stream, &inputs, &dispatch)?)
    }
}
