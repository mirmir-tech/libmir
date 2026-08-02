use super::{TestResult, fixture, validate_format_checkpoint};

#[test]
#[ignore = "MF-140 V2-V4 gate; set MIRMIR_BNB_NF4_MODEL and MIRMIR_BNB_NF4_REFERENCE"]
fn validates_bitsandbytes_nf4_checkpoint_v2_to_v4() -> TestResult<()> {
    validate("MIRMIR_BNB_NF4_MODEL", "MIRMIR_BNB_NF4_REFERENCE")
}

#[test]
#[ignore = "MF-140 V2-V4 gate; set MIRMIR_BNB_FP4_MODEL and MIRMIR_BNB_FP4_REFERENCE"]
fn validates_bitsandbytes_fp4_checkpoint_v2_to_v4() -> TestResult<()> {
    validate("MIRMIR_BNB_FP4_MODEL", "MIRMIR_BNB_FP4_REFERENCE")
}

fn validate(model_env: &str, reference_env: &str) -> TestResult<()> {
    validate_format_checkpoint(
        model_env,
        reference_env,
        fixture::Reference::validate_bitsandbytes_for,
        fixture::validate_bitsandbytes_descriptor,
    )
}
