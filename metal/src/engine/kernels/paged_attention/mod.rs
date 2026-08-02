use super::{Kernels, template};
use crate::engine::{
    Error, Result,
    attention::{PagedAttentionScratch, ScratchSpec},
};
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum PagedExecution {
    Direct,
    TwoPass { blocks: usize, reduction_groups: usize },
}
impl PagedExecution {
    pub const fn blocks(self) -> Option<usize> {
        match self {
            Self::Direct => None,
            Self::TwoPass { blocks, .. } => Some(blocks),
        }
    }
}
#[derive(Clone, Copy)]
struct PagedShape {
    page_size: usize,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    scale: f32,
}
impl Kernels {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn paged_attention(
        &self,
        stream: &mirtal::Stream,
        inputs: [&mirtal::Array; 5],
        scratch: &PagedAttentionScratch,
        page_size: usize,
        context_tokens: usize,
        scale: f32,
        execution: PagedExecution,
    ) -> Result<mirtal::Array> {
        let [queries, key_pages, value_pages, page_table, dependency] = inputs;
        let query = queries.shape()?;
        let keys = key_pages.shape()?;
        validate([&query, &keys, &value_pages.shape()?], inputs, page_size, context_tokens, scale)?;
        let query_heads = query.dimensions()[1];
        let kv_heads = keys.dimensions()[0];
        let head_dim = query.dimensions()[3];
        if let PagedExecution::TwoPass { blocks, reduction_groups } = execution {
            return self.paged_attention_two_pass(
                inputs,
                scratch,
                PagedShape {
                    page_size,
                    query_heads,
                    kv_heads,
                    head_dim,
                    scale,
                },
                blocks,
                reduction_groups,
                stream,
            );
        }
        let scalar = mirtal::Array::from_slice(&[scale], [])?;
        let [output] = self.paged_attention.dispatch(
            stream,
            [queries, key_pages, value_pages, page_table, dependency, &scalar],
            &[mirtal::OutputSpec::new(query, queries.dtype()?)],
            &mirtal::Dispatch::new([1_024, query_heads, 1], [1_024, 1, 1]).templates([
                mirtal::TemplateArg::dtype("T", queries.dtype()?),
                template("QUERY_HEADS", query_heads)?,
                template("KV_HEADS", kv_heads)?,
                template("PAGE_CAPACITY", keys.dimensions()[1])?,
                template("HEAD_DIM", head_dim)?,
                template("QK_PER_THREAD", head_dim.div_ceil(32))?,
                template("V_PER_THREAD", head_dim.div_ceil(32))?,
                template("PAGE_SIZE", page_size)?,
            ]),
        )?;
        Ok(output)
    }

    fn paged_attention_two_pass(
        &self,
        [queries, key_pages, value_pages, page_table, dependency]: [&mirtal::Array; 5],
        scratch: &PagedAttentionScratch,
        shape: PagedShape,
        blocks: usize,
        reduction_groups: usize,
        stream: &mirtal::Stream,
    ) -> Result<mirtal::Array> {
        let PagedShape {
            page_size,
            query_heads,
            kv_heads,
            head_dim,
            scale,
        } = shape;
        let dtype = queries.dtype()?;
        let function = partial_function(head_dim, dtype)?;
        let mut scratch = scratch.lock()?;
        scratch.prepare(
            ScratchSpec {
                query_heads,
                kv_heads,
                page_capacity: key_pages.shape()?.dimensions()[1],
                blocks,
                reduction_groups,
                head_dim,
                page_size,
                scale_bits: scale.to_bits(),
                dtype,
            },
            stream,
            &self.paged_attention_partial,
            function,
        )?;
        let [partials, sums, maximums] = {
            let (kernel, [partial_buffer, sum_buffer, maximum_buffer, barrier]) =
                scratch.partial(dependency)?;
            kernel.dispatch(
                stream,
                [
                    queries, key_pages, value_pages, page_table, dependency, partial_buffer,
                    sum_buffer, maximum_buffer, barrier,
                ],
            )?
        };
        let (reduce_dispatch, reduce_output) = scratch.reduce()?;
        let [output] = self.paged_attention_reduce.dispatch(
            stream,
            [&partials, &sums, &maximums],
            reduce_output,
            reduce_dispatch,
        )?;
        scratch.update([partials, sums, maximums], output.clone());
        drop(scratch);
        Ok(output)
    }
}
pub fn two_pass_supported(
    context_tokens: usize,
    head_dim: usize,
    query_heads: usize,
    kv_heads: usize,
) -> bool {
    kv_heads > 0
        && query_heads > 0
        && query_heads.is_multiple_of(kv_heads)
        && context_tokens >= 1_024
        && head_dim <= 256
        && head_dim.is_multiple_of(32)
        && query_heads / kv_heads <= 32
}
fn partial_function(head_dim: usize, dtype: mirtal::DType) -> Result<&'static str> {
    match (head_dim, dtype) {
        (32, mirtal::DType::Float32) => Ok("mirmir_paged_sdpa_partial_hd32_f32"),
        (64, mirtal::DType::Float32) => Ok("mirmir_paged_sdpa_partial_hd64_f32"),
        (96, mirtal::DType::Float32) => Ok("mirmir_paged_sdpa_partial_hd96_f32"),
        (128, mirtal::DType::Float32) => Ok("mirmir_paged_sdpa_partial_hd128_f32"),
        (160, mirtal::DType::Float32) => Ok("mirmir_paged_sdpa_partial_hd160_f32"),
        (192, mirtal::DType::Float32) => Ok("mirmir_paged_sdpa_partial_hd192_f32"),
        (224, mirtal::DType::Float32) => Ok("mirmir_paged_sdpa_partial_hd224_f32"),
        (256, mirtal::DType::Float32) => Ok("mirmir_paged_sdpa_partial_hd256_f32"),
        (32, mirtal::DType::Float16) => Ok("mirmir_paged_sdpa_partial_hd32_f16"),
        (64, mirtal::DType::Float16) => Ok("mirmir_paged_sdpa_partial_hd64_f16"),
        (96, mirtal::DType::Float16) => Ok("mirmir_paged_sdpa_partial_hd96_f16"),
        (128, mirtal::DType::Float16) => Ok("mirmir_paged_sdpa_partial_hd128_f16"),
        (160, mirtal::DType::Float16) => Ok("mirmir_paged_sdpa_partial_hd160_f16"),
        (192, mirtal::DType::Float16) => Ok("mirmir_paged_sdpa_partial_hd192_f16"),
        (224, mirtal::DType::Float16) => Ok("mirmir_paged_sdpa_partial_hd224_f16"),
        (256, mirtal::DType::Float16) => Ok("mirmir_paged_sdpa_partial_hd256_f16"),
        (32, mirtal::DType::Bfloat16) => Ok("mirmir_paged_sdpa_partial_hd32_bf16"),
        (64, mirtal::DType::Bfloat16) => Ok("mirmir_paged_sdpa_partial_hd64_bf16"),
        (96, mirtal::DType::Bfloat16) => Ok("mirmir_paged_sdpa_partial_hd96_bf16"),
        (128, mirtal::DType::Bfloat16) => Ok("mirmir_paged_sdpa_partial_hd128_bf16"),
        (160, mirtal::DType::Bfloat16) => Ok("mirmir_paged_sdpa_partial_hd160_bf16"),
        (192, mirtal::DType::Bfloat16) => Ok("mirmir_paged_sdpa_partial_hd192_bf16"),
        (224, mirtal::DType::Bfloat16) => Ok("mirmir_paged_sdpa_partial_hd224_bf16"),
        (256, mirtal::DType::Bfloat16) => Ok("mirmir_paged_sdpa_partial_hd256_bf16"),
        _ => Err(Error::InvalidModel("paged attention scratch shape is unsupported".into())),
    }
}

pub(in crate::engine) fn partial_blocks(
    context: usize,
    group_factor: usize,
    kv_heads: usize,
) -> usize {
    let base: usize = if context <= 1_024 || group_factor <= 4 {
        64
    } else if context <= 8_192 {
        128
    } else if context <= 32_768 {
        256
    } else if context <= 65_536 {
        512
    } else {
        1_024
    };
    base.saturating_mul(8_usize.div_ceil(kv_heads)).min(1_024)
}

fn validate(
    shapes: [&mirtal::Shape; 3],
    arrays: [&mirtal::Array; 5],
    page_size: usize,
    context_tokens: usize,
    scale: f32,
) -> Result<()> {
    let [query, keys, values] = shapes.map(mirtal::Shape::dimensions);
    let [queries, key_pages, value_pages, page_table, dependency] = arrays;
    let ranks = query.len() == 4 && keys.len() == 4 && values.len() == 4;
    let layout = ranks
        && query[0] == 1
        && query[2] == 1
        && keys == values
        && keys[0] > 0
        && query[1].is_multiple_of(keys[0])
        && query[3] == keys[3]
        && keys[2] == page_size;
    let dtypes = queries.dtype()? == key_pages.dtype()?
        && key_pages.dtype()? == value_pages.dtype()?
        && page_table.dtype()? == mirtal::DType::Uint32
        && dependency.dtype()? == mirtal::DType::Uint32;
    let pages = context_tokens.div_ceil(page_size.max(1));
    if !layout
        || !dtypes
        || page_size == 0
        || context_tokens == 0
        || query.get(3).is_none_or(|dimension| *dimension > 512)
        || page_table.len() < pages
        || dependency.len() != 1
        || !scale.is_finite()
        || scale <= 0.0
    {
        return Err(Error::InvalidModel("paged SDPA inputs are incompatible".into()));
    }
    Ok(())
}
pub(super) mod batched;
#[cfg(test)]
mod tests;
