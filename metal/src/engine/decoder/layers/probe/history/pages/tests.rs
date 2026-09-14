use super::*;

#[test]
fn page_capture_records_layout_and_cleans_up_scope() -> Result<()> {
    let input = mirtal::Array::from_slice(&[1.0_f32], [1])?;
    let capture = Capture::begin()?;
    assert!(Capture::begin().is_err());
    record(3, &[4, 5], [&input, &input], [&input, &input])?;
    record(7, &[4, 6, 7], [&input, &input], [&input, &input])?;
    let views = capture.finish()?;
    assert!(matches!(views[0].layout, Layout::Contiguous { first: 4, pages: 2 }));
    assert!(matches!(views[1].layout, Layout::Gathered { runs: 2, pages: 3 }));
    drop(Capture::begin()?);
    assert!(Capture::begin()?.finish()?.is_empty());
    Ok(())
}
