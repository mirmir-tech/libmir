use super::{TestResult, fixture, validate_format_checkpoint_for};

#[test]
#[ignore = "mixed ModelOpt V2-V4 gate; set MIRMIR_MODELOPT_MIXED_MODEL and reference"]
fn validates_modelopt_mixed_shared_routed_checkpoint_v2_to_v4() -> TestResult<()> {
    validate_format_checkpoint_for(
        "MIRMIR_MODELOPT_MIXED_MODEL",
        "MIRMIR_MODELOPT_MIXED_REFERENCE",
        fixture::Family::SharedRouted,
        fixture::validate_modelopt_mixed_for,
        fixture::validate_modelopt_descriptor,
    )
}
