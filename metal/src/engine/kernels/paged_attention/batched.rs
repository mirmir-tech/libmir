use super::super::{Kernels, template};
use crate::engine::{Error, Result};

mirtal::metal_kernel! {
    fn paged_attention_batched {
        name: "mirmir_batched_paged_sdpa",
        templates: [
            T: dtype = bf16, QUERY_HEADS: int = 32, KV_HEADS: int = 16,
            PAGE_TABLE_CAPACITY: int = 128, HEAD_DIM: int = 128,
            QK_PER_THREAD: int = 4, V_PER_THREAD: int = 4, PAGE_SIZE: int = 64,
        ],
        inputs: [
            queries: T,
            key_pages_0: T, key_pages_1: T, key_pages_2: T, key_pages_3: T,
            key_pages_4: T, key_pages_5: T, key_pages_6: T, key_pages_7: T,
            value_pages_0: T, value_pages_1: T, value_pages_2: T, value_pages_3: T,
            value_pages_4: T, value_pages_5: T, value_pages_6: T, value_pages_7: T,
            page_tables: u32, page_dependencies: u32, page_capacities: u32,
            attention_scale: scalar<f32>,
        ],
        outputs: [output: T],
        source: file "kernels/paged_attention/batched.metal",
        header: inline "",
        row_contiguous: true,
        atomic_outputs: false,
    }
}

pub(in crate::engine::kernels) fn new() -> Result<mirtal::MetalKernel<21, 1>> {
    Ok(paged_attention_batched()?)
}

impl Kernels {
    pub(crate) fn batched_paged_attention(
        &self,
        stream: &mirtal::Stream,
        inputs: [&mirtal::Array; 20],
        page_size: usize,
        context_tokens: usize,
        scale: f32,
    ) -> Result<mirtal::Array> {
        let query = inputs[0].shape()?;
        let dimensions = query.dimensions();
        let batch = *dimensions.first().ok_or(Error::ShapeOverflow)?;
        let query_heads = *dimensions.get(1).ok_or(Error::ShapeOverflow)?;
        let head_dim = *dimensions.get(3).ok_or(Error::ShapeOverflow)?;
        let keys = inputs[1].shape()?;
        let kv_heads = keys.dimensions()[0];
        let pages = context_tokens.div_ceil(page_size);
        if batch == 0
            || batch > super::super::BATCHED_PAGED_ROWS
            || inputs[17].len() != batch * pages
        {
            return Err(Error::InvalidModel("batched paged SDPA inputs are incompatible".into()));
        }
        let scalar = mirtal::Array::from_slice(&[scale], [])?;
        let kernel_inputs = [
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6], inputs[7],
            inputs[8], inputs[9], inputs[10], inputs[11], inputs[12], inputs[13], inputs[14],
            inputs[15], inputs[16], inputs[17], inputs[18], inputs[19], &scalar,
        ];
        let [output] = self.paged_attention_batched.dispatch(
            stream,
            kernel_inputs,
            &[mirtal::OutputSpec::new(query, inputs[0].dtype()?)],
            &mirtal::Dispatch::new([1_024, query_heads, batch], [1_024, 1, 1]).templates([
                mirtal::TemplateArg::dtype("T", inputs[0].dtype()?),
                template("QUERY_HEADS", query_heads)?,
                template("KV_HEADS", kv_heads)?,
                template("PAGE_TABLE_CAPACITY", pages)?,
                template("HEAD_DIM", head_dim)?,
                template("QK_PER_THREAD", head_dim.div_ceil(32))?,
                template("V_PER_THREAD", head_dim.div_ceil(32))?,
                template("PAGE_SIZE", page_size)?,
            ]),
        )?;
        Ok(output)
    }
}
