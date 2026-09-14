use super::*;
mod cost;

fn fixture(
    stream: &Stream,
    dtype: mirtal::DType,
    shape: [usize; 4],
    tokens: usize,
) -> Result<(History, [Array; 2])> {
    let graph = stream.native().graph();
    let full = mirtal::Shape::new(shape)?;
    let update = mirtal::Shape::new([shape[0], shape[1], 1, shape[3]])?;
    Ok((
        History {
            buffers: [
                Array::from_native(graph.full(&full, 0.0, dtype)?)?,
                Array::from_native(graph.full(&full, 0.0, dtype)?)?,
            ],
            shape,
            tokens,
            reservation: stream
                .history_budget()
                .reserve(crate::engine::persistent_history::budget::storage_bytes(shape, dtype)?)
                .ok_or(Error::HistoryBudgetUnavailable)?,
        },
        [
            Array::from_native(graph.full(&update, 0.5, dtype)?)?,
            Array::from_native(graph.full(&update, -0.25, dtype)?)?,
        ],
    ))
}

#[test]
fn rebind_preserves_pending_dispatch_geometry_and_dtype() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let append = &stream.kernels().history_append;
    let mut pending = Vec::new();
    for _ in 0..2 {
        for dtype in [mirtal::DType::Float32, mirtal::DType::Float16, mirtal::DType::Bfloat16] {
            for (shape, offset) in [([2, 2, 8, 64], 1), ([3, 8, 32, 64], 7), ([1, 2, 16, 256], 11)]
            {
                let (history, updates) = fixture(&stream, dtype, shape, offset)?;
                let outputs = append.execute(&history, [&updates[0], &updates[1]], &stream)?;
                pending.push((shape, offset, outputs));
            }
        }
    }
    assert_eq!(
        append
            .plans
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
            .filter(|p| p.is_some())
            .count(),
        3
    );
    for (shape, offset, outputs) in pending.into_iter().rev() {
        for (output, value) in outputs.into_iter().zip([0.5, -0.25]) {
            let actual = output.to_vec_f32(&stream)?;
            for (index, actual) in actual.into_iter().enumerate() {
                let expected: f32 = if index / shape[3] % shape[2] == offset {
                    value
                } else {
                    0.0
                };
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "shape={shape:?}, offset={offset}, index={index}"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn rejects_invalid_geometry_and_address_overflow() {
    for (shape, offset) in [
        ([0, 2, 16, 64], 0),
        ([3, 2, 16, 64], 16),
        ([3, 2, 16, 64], usize::MAX),
        ([3, 2, usize::MAX, 64], 1),
        ([3, 2, 1 << 24, 64], 1),
    ] {
        assert!(Geometry::new(shape, offset).is_err());
    }
    assert!(Geometry::new([3, 2, 2304, 256], 2241).is_ok());
}
