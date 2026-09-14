use super::*;

#[test]
fn reuses_only_complete_ordered_cohorts() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let rows =
        StateArray::split(Array::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2])?, 3, &stream)?;
    let complete = rows.iter().collect::<Vec<_>>();
    assert!(StateArray::whole_batch(&complete).is_some());
    assert_eq!(
        StateArray::join(&complete, &stream)?.to_vec_f32(&stream)?,
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
    for (cohort, expected) in [
        (vec![&rows[2], &rows[0], &rows[1]], vec![5.0, 6.0, 1.0, 2.0, 3.0, 4.0]),
        (vec![&rows[1]], vec![3.0, 4.0]),
        (vec![&rows[0], &rows[0], &rows[2]], vec![1.0, 2.0, 1.0, 2.0, 5.0, 6.0]),
    ] {
        assert!(StateArray::whole_batch(&cohort).is_none());
        assert_eq!(StateArray::join(&cohort, &stream)?.to_vec_f32(&stream)?, expected);
    }
    Ok(())
}

#[test]
fn snapshots_survive_cohort_advancement_and_graph_detachment() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let mut rows = StateArray::split(Array::from_f32(&[1.0, 2.0], &[2, 1])?, 2, &stream)?;
    let snapshot = rows.iter().map(StateArray::try_clone).collect::<Result<Vec<_>>>()?;
    let next = StateArray::split(Array::from_f32(&[10.0, 20.0], &[2, 1])?, 2, &stream)?;
    rows = next;
    stream.eval_many(&[rows[0].graph_root(), snapshot[0].graph_root()])?;
    stream.synchronize()?;
    rows[0].graph_root().detach_graph(&stream)?;
    snapshot[0].graph_root().detach_graph(&stream)?;
    let mixed = [&rows[0], &snapshot[1]];
    assert!(StateArray::whole_batch(&mixed).is_none());
    assert_eq!(StateArray::join(&mixed, &stream)?.to_vec_f32(&stream)?, vec![10.0, 2.0]);
    let saved = snapshot.iter().collect::<Vec<_>>();
    assert!(StateArray::whole_batch(&saved).is_some());
    assert_eq!(StateArray::join(&saved, &stream)?.to_vec_f32(&stream)?, vec![1.0, 2.0]);
    let owned = StateArray::from(Array::from_f32(&[30.0], &[1, 1])?);
    assert_eq!(
        StateArray::join(&[&snapshot[0], &owned], &stream)?.to_vec_f32(&stream)?,
        vec![1.0, 30.0]
    );
    Ok(())
}

#[test]
fn rejects_incompatible_packed_state_rows() -> Result<()> {
    let stream = Stream::new_gpu()?;
    assert!(StateArray::split(Array::from_f32(&[1.0, 2.0], &[2, 1])?, 3, &stream).is_err());
    assert!(StateArray::split(Array::from_f32(&[1.0], &[])?, 1, &stream).is_err());
    Ok(())
}
