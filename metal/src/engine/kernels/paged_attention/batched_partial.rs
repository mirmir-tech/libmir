use super::super::{Kernels, template};
use crate::engine::{Error, Result};

mirtal::metal_kernel! {
    fn paged_attention_batched_partial {
        name: "mirmir_batched_paged_sdpa_partial",
        templates: [
            T: dtype = bf16, BATCH: int = 2, QUERY_HEADS: int = 32, KV_HEADS: int = 16,
            HEAD_DIM: int = 128,
            BLOCKS: int = 256, SCALE_BITS: int = 1040187392, PAGE_SIZE: int = 64,
        ],
        inputs: [
            queries: T,
            key_pages_0: T, key_pages_1: T, key_pages_2: T, key_pages_3: T,
            key_pages_4: T, key_pages_5: T, key_pages_6: T, key_pages_7: T,
            key_pages_8: T, key_pages_9: T, key_pages_10: T, key_pages_11: T,
            value_pages_0: T, value_pages_1: T, value_pages_2: T, value_pages_3: T,
            value_pages_4: T, value_pages_5: T, value_pages_6: T, value_pages_7: T,
            value_pages_8: T, value_pages_9: T, value_pages_10: T, value_pages_11: T,
            page_tables: u32, metadata: u32,
        ],
        outputs: [partials: T, sums: f32, maximums: f32],
        source: file "kernels/paged_attention/batched_partial.metal",
        header: file "kernels/paged_attention/partial.metal",
        row_contiguous: true,
        atomic_outputs: false,
    }
}

pub(in crate::engine::kernels) fn new() -> Result<mirtal::MetalKernel<27, 3>> {
    Ok(paged_attention_batched_partial()?)
}

impl Kernels {
    pub(crate) fn batched_paged_two_pass(
        &self,
        stream: &mirtal::Stream,
        inputs: [&mirtal::Array; 28],
        page_size: usize,
        context_tokens: usize,
        scale: f32,
    ) -> Result<mirtal::Array> {
        let shape = inputs[0].shape()?;
        let dimensions = shape.dimensions();
        let [batch, heads, sequence, head_dim] = dimensions else {
            return Err(Error::ShapeOverflow);
        };
        let kv_heads = inputs[1].shape()?.dimensions()[0];
        if *batch == 0
            || *batch > 12
            || *sequence != 1
            || !super::two_pass_supported(context_tokens, *head_dim, *heads, kv_heads)
        {
            return Err(Error::InvalidModel("unsupported two-pass paged batch".into()));
        }
        let blocks = super::partial_blocks(context_tokens, heads / kv_heads, kv_heads);
        let pages = context_tokens.div_ceil(page_size);
        if inputs[25].len() != batch * pages {
            return Err(Error::ShapeOverflow);
        }
        let dtype = inputs[0].dtype()?;
        let statistic = mirtal::Shape::new([*batch, *heads, 1, blocks])?;
        // The used page count changes during decode. Keep the table stride in
        // runtime metadata so crossing a page never specializes a new pipeline.
        let table_stride = mirtal::Array::from_slice(&[u32::try_from(pages)?], [1])?;
        let metadata = stream.graph().concatenate(&[inputs[26], inputs[27], &table_stride], 0)?;
        let kernel_inputs = std::array::from_fn(|index| {
            if index == 26 {
                &metadata
            } else {
                inputs[index]
            }
        });
        let [partials, sums, maximums] = self.paged_attention_batched_partial.dispatch(
            stream,
            kernel_inputs,
            &[
                mirtal::OutputSpec::new(
                    mirtal::Shape::new([*batch, *heads, 1, blocks, *head_dim])?,
                    dtype,
                ),
                mirtal::OutputSpec::new(statistic.clone(), mirtal::DType::Float32),
                mirtal::OutputSpec::new(statistic, mirtal::DType::Float32),
            ],
            &mirtal::Dispatch::new([32, *heads, blocks * batch], [32, heads / kv_heads, 1])
                .templates([
                    mirtal::TemplateArg::dtype("T", dtype),
                    template("BATCH", *batch)?,
                    template("QUERY_HEADS", *heads)?,
                    template("KV_HEADS", kv_heads)?,
                    template("HEAD_DIM", *head_dim)?,
                    template("PAGE_SIZE", page_size)?,
                    template("BLOCKS", blocks)?,
                    mirtal::TemplateArg::int(
                        "SCALE_BITS",
                        i32::from_ne_bytes(scale.to_bits().to_ne_bytes()),
                    ),
                ]),
        )?;
        let [output] = self.paged_attention_reduce.dispatch(
            stream,
            [&partials, &sums, &maximums],
            &[mirtal::OutputSpec::new(shape.clone(), dtype)],
            &mirtal::Dispatch::new([1024, heads * batch, 1], [1024, 1, 1]).templates([
                mirtal::TemplateArg::dtype("T", dtype),
                template("HEAD_DIM", *head_dim)?,
                template("BLOCKS", blocks)?,
                template("REDUCTION_GROUPS", 32)?,
            ]),
        )?;
        Ok(output)
    }
}
