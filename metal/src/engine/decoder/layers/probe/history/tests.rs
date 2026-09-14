use super::*;
#[test]
fn history_capture_tracks_layers_and_scope_cleanup() -> Result<()> {
    let input = Array::from_f32(&[1.0], &[1, 1, 1, 1])?;
    row(&input, &input, &input, Mask::None, Bias::None, Reader::RowView)?;
    let capture = Capture::begin()?;
    assert!(Capture::begin().is_err());
    assert!(row(&input, &input, &input, Mask::None, Bias::None, Reader::RowView).is_err());
    enter_layer(3);
    row(&input, &input, &input, Mask::Explicit, Bias::Sinks, Reader::RowView)?;
    let (records, replays) = capture.finish()?;
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].layer, 3);
    assert!(replays.is_empty());
    drop(Capture::begin()?);
    assert!(Capture::begin()?.finish()?.0.is_empty());
    Ok(())
}
