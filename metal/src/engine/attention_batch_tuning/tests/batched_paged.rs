use super::{BatchAttentionExecution, execute, support};
use crate::engine::{Array, Result, Stream};

#[test]
fn executes_ten_paged_rows_in_one_batch() -> Result<()> {
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
    let actual = execute(
        BatchAttentionExecution::PagedBatched,
        &queries,
        &contexts,
        0.125,
        false,
        &stream,
    )?;
    support::assert_outputs_close(&expected, &actual, &stream)
}
