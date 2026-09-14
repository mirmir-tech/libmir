use super::*;

#[test]
fn captures_layer_indices_and_recovers_after_drop() -> Result<()> {
    let indices = Array::from_u32(&[2, 5, 7, 2], &[2, 2])?;
    record(&indices)?;
    let capture = Capture::begin()?;
    assert!(Capture::begin().is_err());
    assert!(record(&indices).is_err());
    enter_layer(7);
    record(&indices)?;
    enter_layer(11);
    record(&indices)?;
    let routes = capture.finish()?;
    assert_eq!(routes.iter().map(|r| r.layer).collect::<Vec<_>>(), [7, 11]);
    assert_eq!(routes[0].indices.shape()?, [2, 2]);
    drop(Capture::begin()?);
    assert!(Capture::begin()?.finish()?.is_empty());
    Ok(())
}
