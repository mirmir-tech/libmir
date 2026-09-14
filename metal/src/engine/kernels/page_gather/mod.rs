use crate::engine::{Error, Result, Stream};

mirtal::metal_kernel! {
    fn page_gather {
        name: "mirmir_page_gather_batch",
        templates: [T: dtype = bf16, HEAD_DIM: int = 256, KV_HEADS: int = 2, PAGE_SIZE: int = 16],
        inputs: [
            key_pages_0: T, key_pages_1: T, key_pages_2: T, key_pages_3: T,
            key_pages_4: T, key_pages_5: T, key_pages_6: T, key_pages_7: T,
            key_pages_8: T, key_pages_9: T, key_pages_10: T, key_pages_11: T,
            value_pages_0: T, value_pages_1: T, value_pages_2: T, value_pages_3: T,
            value_pages_4: T, value_pages_5: T, value_pages_6: T, value_pages_7: T,
            value_pages_8: T, value_pages_9: T, value_pages_10: T, value_pages_11: T,
            tables: u32, metadata: u32,
        ],
        outputs: [keys: T, values: T],
        source: file "kernels/page_gather.metal",
        header: inline "",
        row_contiguous: true,
        atomic_outputs: false,
    }
}

pub struct Source<'a> {
    pub keys: &'a mirtal::Array,
    pub values: &'a mirtal::Array,
    pub table: &'a mirtal::Array,
}

#[derive(Debug)]
pub struct Gather {
    kernel: mirtal::MetalKernel<26, 2>,
}

impl Gather {
    pub(crate) fn new() -> Result<Self> {
        Ok(Self { kernel: page_gather()? })
    }

    // Page IDs are owned/validated by the cache. Their contents stay on device.
    pub(crate) fn execute(
        &self,
        sources: &[Source<'_>],
        tokens: usize,
        stream: &Stream,
    ) -> Result<[mirtal::Array; 2]> {
        let Some(first) = sources.first() else {
            return Err(incompatible());
        };
        if sources.len() > 12 || tokens == 0 {
            return Err(incompatible());
        }
        let shape = first.keys.shape()?;
        let &[heads, _, page_size, dim] = shape.dimensions() else {
            return Err(incompatible());
        };
        if heads == 0 || page_size == 0 || dim == 0 {
            return Err(incompatible());
        }
        let dtype = first.keys.dtype()?;
        if !matches!(
            dtype,
            mirtal::DType::Float32 | mirtal::DType::Float16 | mirtal::DType::Bfloat16
        ) {
            return Err(incompatible());
        }
        let pages = tokens.div_ceil(page_size);
        let mut metadata = vec![
            u32::try_from(tokens)?,
            u32::try_from(heads)?,
            u32::try_from(dim)?,
            u32::try_from(page_size)?,
            u32::try_from(pages)?,
            u32::try_from(sources.len())?,
        ];
        let graph = stream.native().graph();
        let mut tables = Vec::new();
        for source in sources {
            let shape = source.keys.shape()?;
            let &[h, capacity, s, d] = shape.dimensions() else {
                return Err(incompatible());
            };
            if [h, s, d] != [heads, page_size, dim]
                || capacity == 0
                || source.values.shape()? != shape
                || source.keys.dtype()? != dtype
                || source.values.dtype()? != dtype
                || source.table.dtype()? != mirtal::DType::Uint32
                || source.table.shape()?.dimensions().len() != 1
                || source.table.len() < pages
            {
                return Err(incompatible());
            }
            let _extent = u32::try_from(source.keys.len())?;
            metadata.push(u32::try_from(capacity)?);
            tables.push(graph.slice(source.table, &[0], &[pages])?);
        }
        let tables = graph.concatenate(&tables.iter().collect::<Vec<_>>(), 0)?;
        let metadata = mirtal::Array::from_slice(&metadata, [metadata.len()])?;
        let output_shape = mirtal::Shape::new([sources.len(), heads, tokens, dim])?;
        let elements = sources
            .len()
            .checked_mul(heads)
            .and_then(|n| n.checked_mul(tokens))
            .and_then(|n| n.checked_mul(dim))
            .ok_or(Error::ShapeOverflow)?;
        let _extent = u32::try_from(elements)?;
        let inputs = std::array::from_fn(|index| match index {
            24 => &tables,
            25 => &metadata,
            0..12 => sources.get(index).unwrap_or(first).keys,
            _ => sources.get(index - 12).unwrap_or(first).values,
        });
        let output = mirtal::OutputSpec::new(output_shape, dtype);
        Ok(self.kernel.dispatch(
            stream.native(),
            inputs,
            &[output.clone(), output],
            &mirtal::Dispatch::new(
                [dim.div_ceil(4), tokens, heads * sources.len()],
                [dim.div_ceil(4).min(256), 1, 1],
            )
            .templates([
                mirtal::TemplateArg::dtype("T", dtype),
                super::template("HEAD_DIM", dim)?,
                super::template("KV_HEADS", heads)?,
                super::template("PAGE_SIZE", page_size)?,
            ]),
        )?)
    }
}

fn incompatible() -> Error {
    Error::InvalidModel("incompatible page gather sources or extent".into())
}

#[cfg(test)]
mod tests;
