use std::sync::Arc;

use crate::engine::{Array, KvCache, KvPageFormat, PagedContextMode, Result, Stream};

const ROWS: usize = 3;
const HEADS: usize = 2;
const DIM: usize = 8;

fn tokens(step: usize, rows: usize) -> Result<Array> {
    let values = (0..rows * HEADS * DIM)
        .map(|i| Ok(f32::from(u16::try_from((i * 7 + step * 13) % 31)?) / 8.0))
        .collect::<Result<Vec<_>>>()?;
    let shape = [i32::try_from(rows)?, i32::try_from(HEADS)?, 1, i32::try_from(DIM)?];
    Array::from_f32(&values, &shape)
}

fn row(array: &Array, index: usize, stream: &Stream) -> Result<Array> {
    array.slice(&[index, 0, 0, 0], &[index + 1, HEADS, 1, DIM], stream)
}

/// Row writes take host slots and skip the page table, so a table that grew
/// meanwhile must survive a flush that detaches graphs, as native reads need
/// it.
#[test]
fn row_writes_match_row_updates_and_keep_page_tables_after_flushes() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let pool = Arc::clone(stream.paged_arenas());
    let paged =
        |layer| KvCache::new_paged_with_pool(16, 4, KvPageFormat::Native, Arc::clone(&pool), layer);
    let mut written = (0..ROWS).map(|_| paged(0)).collect::<Result<Vec<_>>>()?;
    let mut reference = (0..ROWS).map(|_| paged(1)).collect::<Result<Vec<_>>>()?;
    for step in 0..14 {
        let update = tokens(step, ROWS)?;
        if step == 0 {
            for (index, cache) in written.iter_mut().enumerate() {
                let token = row(&update, index, &stream)?;
                cache.update_for_attention_mode(
                    &token,
                    &token,
                    &stream,
                    0,
                    PagedContextMode::View,
                )?;
            }
        } else {
            let mut rows = written.iter_mut().collect::<Vec<_>>();
            assert!(KvCache::write_rows(&mut rows, [&update, &update], &stream)?);
        }
        for (index, cache) in reference.iter_mut().enumerate() {
            let token = row(&update, index, &stream)?;
            cache.update_for_attention_mode(&token, &token, &stream, 0, PagedContextMode::View)?;
        }
        stream.eval_many_with_paged_arenas(&[])?;
        stream.synchronize()?;
        stream.detach_paged_arena_graphs()?;
        for cache in written.iter().chain(&reference) {
            cache.detach_evaluated_graphs(&stream)?;
        }
    }
    let probe = tokens(99, 1)?;
    for (cache, expected) in written.iter_mut().zip(reference.iter_mut()) {
        let native = cache.update_for_attention_mode(
            &probe,
            &probe,
            &stream,
            0,
            PagedContextMode::Native,
        )?;
        let table = native.paged.as_ref().map(|paged| &paged.page_table);
        stream.eval_many(&table.into_iter().collect::<Vec<_>>())?;
        let view =
            cache.update_for_attention_mode(&probe, &probe, &stream, 0, PagedContextMode::View)?;
        expected.update_for_attention_mode(&probe, &probe, &stream, 0, PagedContextMode::View)?;
        let wanted = expected.update_for_attention_mode(
            &probe,
            &probe,
            &stream,
            0,
            PagedContextMode::View,
        )?;
        assert_eq!(view.keys.to_vec_f32(&stream)?, wanted.keys.to_vec_f32(&stream)?);
    }
    Ok(())
}
