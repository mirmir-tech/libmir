use super::{TestResult, fixture, validate_format_checkpoint_for};

#[test]
#[ignore = "affine dense-and-routed V2-V4 gate; set model and reference variables"]
fn validates_affine_dense_and_routed_checkpoint_v2_to_v4() -> TestResult<()> {
    validate_format_checkpoint_for(
        "MIRMIR_AFFINE_DENSE_ROUTED_MODEL",
        "MIRMIR_AFFINE_DENSE_ROUTED_REFERENCE",
        fixture::Family::DenseAndRouted,
        fixture::Reference::validate_affine_for,
        fixture::validate_affine_descriptor,
    )
}
