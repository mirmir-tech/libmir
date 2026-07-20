use super::QuantizedKvKernels;
use crate::engine::{Error, Result};

impl QuantizedKvKernels {
    pub(super) fn attention(
        &self,
        stream: &mirtal::Stream,
        inputs: [&mirtal::Array; 7],
        page_size: usize,
        context_tokens: usize,
        scale: f32,
    ) -> Result<mirtal::Array> {
        let [query, keys, values, key_scales, value_scales, page_table, dependency] = inputs;
        let query_shape = query.shape()?;
        let key_shape = keys.shape()?;
        validate(
            [&query_shape, &key_shape, &values.shape()?],
            [&key_scales.shape()?, &value_scales.shape()?],
            inputs,
            page_size,
            context_tokens,
            scale,
        )?;
        let query_heads = query_shape.dimensions()[1];
        let query_tokens = query_shape.dimensions()[2];
        let kv_heads = key_shape.dimensions()[0];
        let head_dim = query_shape.dimensions()[3];
        let scalar = mirtal::Array::from_slice(&[scale], [])?;
        let [output] = self.attention.dispatch(
            stream,
            [query, keys, values, key_scales, value_scales, page_table, dependency, &scalar],
            &[mirtal::OutputSpec::new(query_shape, query.dtype()?)],
            &mirtal::Dispatch::new([1_024, query_heads, query_tokens], [1_024, 1, 1]).templates([
                mirtal::TemplateArg::dtype("T", query.dtype()?),
                template("QUERY_HEADS", query_heads)?,
                template("KV_HEADS", kv_heads)?,
                template("QUERY_TOKENS", query_tokens)?,
                template("PAGE_CAPACITY", key_shape.dimensions()[1])?,
                template("HEAD_DIM", head_dim)?,
                template("PACKED_DIM", key_shape.dimensions()[3])?,
                template("QK_PER_THREAD", head_dim.div_ceil(32))?,
                template("V_PER_THREAD", head_dim.div_ceil(32))?,
                template("PAGE_SIZE", page_size)?,
            ]),
        )?;
        Ok(output)
    }
}

fn template(name: &'static str, value: usize) -> Result<mirtal::TemplateArg> {
    Ok(mirtal::TemplateArg::int(name, i32::try_from(value)?))
}

fn validate(
    shapes: [&mirtal::Shape; 3],
    scale_shapes: [&mirtal::Shape; 2],
    arrays: [&mirtal::Array; 7],
    page_size: usize,
    context_tokens: usize,
    scale: f32,
) -> Result<()> {
    let [query, keys, values] = shapes.map(mirtal::Shape::dimensions);
    let [key_scale, value_scale] = scale_shapes.map(mirtal::Shape::dimensions);
    let [queries, key_pages, value_pages, key_scales, value_scales, page_table, dependency] =
        arrays;
    let packed = query.get(3).map(|dimension| dimension.div_ceil(4));
    let layout = query.len() == 4
        && keys.len() == 4
        && values == keys
        && query[0] == 1
        && query[2] > 0
        && keys[0] > 0
        && query[1].is_multiple_of(keys[0])
        && packed == keys.get(3).copied()
        && key_scale == &keys[..3]
        && value_scale == key_scale;
    let dtypes = matches!(
        queries.dtype()?,
        mirtal::DType::Float32 | mirtal::DType::Float16 | mirtal::DType::Bfloat16
    ) && key_pages.dtype()? == mirtal::DType::Uint32
        && value_pages.dtype()? == mirtal::DType::Uint32
        && key_scales.dtype()? == mirtal::DType::Float32
        && value_scales.dtype()? == mirtal::DType::Float32
        && page_table.dtype()? == mirtal::DType::Uint32
        && dependency.dtype()? == mirtal::DType::Uint32;
    let pages = context_tokens.div_ceil(page_size.max(1));
    if !layout
        || !dtypes
        || page_size == 0
        || context_tokens < query[2]
        || query[3] > 512
        || page_table.len() < pages
        || dependency.len() != 1
        || !scale.is_finite()
        || scale <= 0.0
    {
        return Err(Error::InvalidModel("INT8 paged SDPA inputs are incompatible".into()));
    }
    Ok(())
}
