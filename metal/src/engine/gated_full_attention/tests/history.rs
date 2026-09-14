use super::*;
use crate::config::HistoryBatching::{Joined, Persistent};

#[test]
fn persistent_decode_matches_joined_through_cohort_churn_and_ragged_fallback() -> Result<()> {
    let mut stream = Stream::new_gpu()?;
    let attention = fixture(&stream)?;
    let mut baseline = (0..3).map(|_| KvCache::new(16)).collect::<Result<Vec<_>>>()?;
    let mut candidate = (0..3).map(|_| KvCache::new(16)).collect::<Result<Vec<_>>>()?;
    let before = crate::engine::persistent_history::counts();
    for step in 0..16 {
        for caches in [&mut baseline, &mut candidate] {
            match step {
                3 => caches.swap(0, 2),
                5 => {
                    caches.remove(1);
                },
                7 => caches.push(caches[0].snapshot_at(caches[0].offset()?)?),
                9 => caches[1] = caches[1].snapshot_at(caches[1].offset()?)?,
                10 => {
                    for cache in caches {
                        cache.release_reservation_after(cache.offset()?)?;
                    }
                },
                11 => {
                    let position = i32::try_from(caches[0].offset()?)?;
                    let input = Array::from_f32(&[0.35; 64], &[1, 1, 64])?;
                    let output =
                        attention.forward(&input, &mut caches[0], 0, position, false, &stream)?;
                    output.async_eval(&stream)?;
                },
                _ => {},
            }
        }
        let width = baseline.len();
        let data = (0..width * 64)
            .map(|i| Ok(f32::from(u16::try_from((i + step) % 17)?) / 32.0))
            .collect::<Result<Vec<_>>>()?;
        let input = Array::from_f32(&data, &[i32::try_from(width)?, 1, 64])?;
        let mut outputs = Vec::new();
        for (plan, caches) in [(Joined, &mut baseline), (Persistent, &mut candidate)] {
            stream.set_history_batching(plan);
            let positions = caches
                .iter()
                .map(|c| Ok(i32::try_from(c.offset()?)?))
                .collect::<Result<Vec<_>>>()?;
            outputs.push(
                attention
                    .forward_packed_decode_with_mode(
                        &input,
                        &mut caches.iter_mut().collect::<Vec<_>>(),
                        &positions,
                        PagedContextMode::View,
                        &stream,
                    )?
                    .to_vec_f32(&stream)?,
            );
        }
        assert_eq!(outputs[0], outputs[1], "cohort churn step {step}");
    }
    let after = crate::engine::persistent_history::counts();
    assert_eq!(after[0] - before[0], 6);
    assert_eq!(after[1] - before[1], 5);
    Ok(())
}
