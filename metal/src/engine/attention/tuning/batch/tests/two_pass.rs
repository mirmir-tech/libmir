use super::{
    Array, BatchAttentionExecution, KvCache, KvContext, PagedContextMode, Result, Stream, execute,
    patterned,
};
use crate::engine::Dtype;

#[test]
fn two_pass_batch_preserves_gqa_row_arithmetic_and_lengths() -> Result<()> {
    let mut config = crate::MetalConfig::default();
    config.tuning.mode = runtime::tuning::TuningMode::Disabled;
    let stream = Stream::new_gpu_with_config(std::sync::Arc::new(config))?;
    for dtype in [Dtype::Float32, Dtype::Bfloat16] {
        let contexts = [context(1025, 7, dtype, &stream)?, context(1033, 13, dtype, &stream)?];
        let queries = [query(19, dtype, &stream)?, query(23, dtype, &stream)?];
        let contexts = contexts.iter().collect::<Vec<_>>();
        let queries = queries.iter().collect::<Vec<_>>();
        let rows = execute(
            BatchAttentionExecution::PagedRows,
            &queries,
            &contexts,
            0.0625,
            false,
            &stream,
        )?;
        for row in &rows {
            drop(row.to_vec_f32(&stream)?);
        }
        let batch = execute(
            BatchAttentionExecution::PagedBatchedTwoPass,
            &queries,
            &contexts,
            0.0625,
            false,
            &stream,
        )?;
        for (row, actual) in rows.iter().zip(&batch) {
            assert_eq!(row.to_vec_f32(&stream)?, actual.to_vec_f32(&stream)?, "dtype={dtype:?}");
        }
    }
    Ok(())
}

fn query(seed: usize, dtype: Dtype, stream: &Stream) -> Result<Array> {
    let values = (0..16 * 256).map(|index| patterned(index, seed)).collect::<Vec<_>>();
    Array::from_f32(&values, &[1, 16, 1, 256])?.astype(dtype, stream)
}

fn context(tokens: usize, seed: usize, dtype: Dtype, stream: &Stream) -> Result<KvContext> {
    let values = (0..2 * (tokens - 1) * 256)
        .map(|index| patterned(index, seed))
        .collect::<Vec<_>>();
    let keys = Array::from_f32(&values, &[1, 2, i32::try_from(tokens - 1)?, 256])?
        .astype(dtype, stream)?;
    let mut cache = KvCache::new_paged(tokens, 16)?;
    drop(cache.update_for_attention_mode(&keys, &keys, stream, 0, PagedContextMode::View)?);
    let token = Array::from_f32(&[0.3; 512], &[1, 2, 1, 256])?.astype(dtype, stream)?;
    cache.update_for_attention_mode(&token, &token, stream, 0, PagedContextMode::Native)
}
