use super::*;

#[test]
fn explicit_policy_admits_compressed_weights_for_prepared_decode_roles() -> Result<()> {
    for (role, input_features, output_features) in [
        (DenseRole::AttentionOutput, 4_096, 2_560),
        (DenseRole::DenseGateUp, 2_560, 19_456),
        (DenseRole::DenseDown, 9_728, 2_560),
    ] {
        let request = DensePlanRequest {
            phase: ExecutionPhase::Decode,
            role,
            tokens: 1,
            input_features,
            output_features,
        };
        assert_weight_plan(
            request,
            CudaDenseWeightPolicy::BlockFp8Role(role),
            DenseExecution::BlockFp8Vector,
        )?;
        assert_weight_plan(
            request,
            CudaDenseWeightPolicy::Fp8Int4Role(role),
            DenseExecution::Fp8Int4Vector,
        )?;
        assert_weight_plan(
            DensePlanRequest {
                phase: ExecutionPhase::Prefill,
                ..request
            },
            CudaDenseWeightPolicy::BlockFp8Role(role),
            DenseExecution::Matrix,
        )?;
        for (rejected, expected) in [
            (DensePlanRequest { tokens: 2, ..request }, DenseExecution::Matrix),
            (
                DensePlanRequest {
                    input_features: input_features + 1,
                    ..request
                },
                DenseExecution::Matrix,
            ),
            (
                DensePlanRequest {
                    input_features: 2_048,
                    output_features: 512,
                    ..request
                },
                DenseExecution::BlockFp8Vector,
            ),
        ] {
            assert_weight_plan(rejected, CudaDenseWeightPolicy::BlockFp8Role(role), expected)?;
        }
    }
    Ok(())
}

fn assert_weight_plan(
    request: DensePlanRequest,
    dense_weights: CudaDenseWeightPolicy,
    expected: DenseExecution,
) -> Result<()> {
    let weighted_planner = planner_with_policy(
        12,
        CudaPlanningPolicy {
            numerical: CudaNumericalPolicy::Throughput,
            admission: CudaKernelAdmission::Experimental,
            dense_weights,
            ..CudaPlanningPolicy::default()
        },
    )?;
    assert_eq!(
        weighted_planner.plan_dense_with_prepared_weights(request)?.execution(),
        expected
    );
    let without_weights = planner(12)?.plan_dense(request)?.execution();
    assert_eq!(weighted_planner.plan_dense(request)?.execution(), without_weights);
    Ok(())
}
