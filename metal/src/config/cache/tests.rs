use super::DecodeReservation;

#[test]
fn reservation_policy_has_one_stable_external_spelling() -> serde_json::Result<()> {
    for policy in [DecodeReservation::OnePage, DecodeReservation::GenerationBudget] {
        let value = serde_json::to_value(policy)?;
        assert_eq!(value.as_str(), Some(policy.to_string().as_str()));
        assert_eq!(serde_json::from_value::<DecodeReservation>(value)?, policy);
    }
    assert_eq!(DecodeReservation::default(), DecodeReservation::OnePage);
    Ok(())
}

#[test]
fn reservation_configuration_rejects_unrecognized_and_test_only_values() {
    for input in ["\"auto\"", "\"unbounded\"", "\"GenerationBudget\"", "256", "{\"tokens\":256}"] {
        assert!(serde_json::from_str::<DecodeReservation>(input).is_err());
    }
}
