use super::*;

#[test]
fn derives_a_stable_model_identifier() -> Result<()> {
    assert_eq!(model_id(Path::new("/models/example"))?, "example");
    Ok(())
}

#[test]
fn rejects_a_request_larger_than_the_model_context() {
    assert!(matches!(
        validate_context(900, 200, 1_024),
        Err(Error::Context { requested: 1_100, .. })
    ));
}
