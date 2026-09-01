use super::*;

#[test]
fn auto_selects_validated_refinement_and_gates_experiments() -> Result<()> {
    let request = OutputHeadPlanRequest {
        input_features: 2_816,
        output_features: 262_144,
    };
    assert_eq!(
        planner(12)?.plan_output_head(request)?.execution(),
        OutputHeadExecution::AutoRefined
    );
    assert_eq!(
        planner(12)?
            .plan_output_head(OutputHeadPlanRequest {
                input_features: 4_096,
                output_features: 151_936,
            })?
            .execution(),
        OutputHeadExecution::AutoRefined
    );
    let gpt_oss = OutputHeadPlanRequest {
        input_features: 2_880,
        output_features: 201_088,
    };
    assert_eq!(planner(12)?.plan_output_head(gpt_oss)?.execution(), OutputHeadExecution::Bf16);
    assert_eq!(
        planner_with_policy(12, experimental(CudaOutputHeadPolicy::Auto))?
            .plan_output_head(gpt_oss)?
            .execution(),
        OutputHeadExecution::Fp8BlockRefined
    );
    for request in [
        OutputHeadPlanRequest {
            input_features: 2_048,
            output_features: 248_320,
        },
        OutputHeadPlanRequest {
            input_features: 5_120,
            output_features: 248_320,
        },
    ] {
        assert_eq!(
            planner_with_policy(12, experimental(CudaOutputHeadPolicy::Auto))?
                .plan_output_head(request)?
                .execution(),
            OutputHeadExecution::Fp8BlockRefined
        );
    }
    for request in [
        OutputHeadPlanRequest {
            input_features: 2_016,
            output_features: 248_320,
        },
        OutputHeadPlanRequest {
            input_features: 2_048,
            output_features: 131_071,
        },
    ] {
        assert_eq!(
            planner_with_policy(12, experimental(CudaOutputHeadPolicy::Auto))?
                .plan_output_head(request)?
                .execution(),
            OutputHeadExecution::Bf16
        );
    }
    assert_execution(request, CudaOutputHeadPolicy::Bf16, OutputHeadExecution::Bf16)?;
    for (policy, execution) in [
        (CudaOutputHeadPolicy::Fp8Blockwise, OutputHeadExecution::Fp8Blockwise),
        (CudaOutputHeadPolicy::Fp8Vectorized, OutputHeadExecution::Fp8Vectorized),
        (CudaOutputHeadPolicy::Fp8Residual, OutputHeadExecution::Fp8Residual),
        (
            CudaOutputHeadPolicy::Fp8BlockVectorized,
            OutputHeadExecution::Fp8BlockVectorized,
        ),
        (CudaOutputHeadPolicy::Fp8BlockRefined, OutputHeadExecution::Fp8BlockRefined),
    ] {
        assert_execution(request, policy, execution)?;
    }
    let selected = experimental(CudaOutputHeadPolicy::Fp8Blockwise);
    assert_eq!(
        planner_with_policy(11, selected)?.plan_output_head(request)?.execution(),
        OutputHeadExecution::Bf16
    );
    Ok(())
}

fn assert_execution(
    request: OutputHeadPlanRequest,
    policy: CudaOutputHeadPolicy,
    expected: OutputHeadExecution,
) -> Result<()> {
    let planning = if policy == CudaOutputHeadPolicy::Bf16 {
        CudaPlanningPolicy {
            output_head: policy,
            ..CudaPlanningPolicy::default()
        }
    } else {
        experimental(policy)
    };
    assert_eq!(
        planner_with_policy(12, planning)?.plan_output_head(request)?.execution(),
        expected
    );
    Ok(())
}

fn experimental(output_head: CudaOutputHeadPolicy) -> CudaPlanningPolicy {
    CudaPlanningPolicy {
        numerical: CudaNumericalPolicy::Throughput,
        admission: CudaKernelAdmission::Experimental,
        output_head,
        ..CudaPlanningPolicy::default()
    }
}
