use super::{BatchAttentionExecution, candidates, execute, support};
use crate::engine::{Array, Result, Stream, attention_batch_tuning::BatchAttentionKey};

#[test]
fn exposes_all_chunk_widths_to_the_tuner_at_batch_ten() {
    let key = BatchAttentionKey {
        batch: 10,
        sequence: 1,
        context_bucket: 2_048,
        query_heads: 16,
        kv_heads: 8,
        head_dim: 256,
        dtype: 5,
        causal: false,
        fragmented: true,
        view: false,
    };
    assert_eq!(
        candidates(key, true, true),
        [
            BatchAttentionExecution::PagedRows,
            BatchAttentionExecution::PagedBatched4,
            BatchAttentionExecution::PagedBatched8,
            BatchAttentionExecution::PagedBatched12,
        ]
    );
}

#[test]
fn executes_ten_paged_rows_with_each_chunk_width() -> Result<()> {
    const ROWS: usize = 10;
    const CONTEXT: usize = 257;
    const HEAD_DIM: usize = 32;

    let stream = Stream::new_gpu()?;
    let query = Array::from_f32(&[0.25; HEAD_DIM], &[1, 1, 1, i32::try_from(HEAD_DIM)?])?;
    let contexts = (0..ROWS)
        .map(|row| support::native_decode_context(CONTEXT, HEAD_DIM, row + 11, &stream))
        .collect::<Result<Vec<_>>>()?;
    let queries = std::iter::repeat_n(&query, ROWS).collect::<Vec<_>>();
    let contexts = contexts.iter().collect::<Vec<_>>();
    let expected =
        execute(BatchAttentionExecution::PagedRows, &queries, &contexts, 0.125, false, &stream)?;
    for execution in [
        BatchAttentionExecution::PagedBatched4,
        BatchAttentionExecution::PagedBatched8,
        BatchAttentionExecution::PagedBatched12,
    ] {
        let actual = execute(execution, &queries, &contexts, 0.125, false, &stream)?;
        support::assert_outputs_close(&expected, &actual, &stream)?;
    }
    Ok(())
}
