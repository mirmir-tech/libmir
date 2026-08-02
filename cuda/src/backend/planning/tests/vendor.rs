use super::*;

#[test]
fn automatic_prefill_keeps_safe_fallback_for_runtime_tuning() -> Result<()> {
    let request = DensePlanRequest {
        phase: ExecutionPhase::Prefill,
        role: DenseRole::AttentionQkv,
        tokens: 256,
        input_features: 3_072,
        output_features: 7_168,
    };
    for major in [11, 12] {
        let plan = planner(major)?.plan_dense(request)?;
        assert_eq!(plan.execution(), DenseExecution::Matrix);
        assert_eq!(plan.source(), PlanSource::Fallback);
    }
    Ok(())
}

#[test]
fn sm12_decode_heuristic_depends_on_operation_geometry() -> Result<()> {
    let sm12 = planner(12)?;
    let request = DensePlanRequest {
        phase: ExecutionPhase::Decode,
        role: DenseRole::AttentionQkv,
        tokens: 1,
        input_features: 3_072,
        output_features: 7_168,
    };
    for role in [
        DenseRole::AttentionQkv,
        DenseRole::AttentionOutput,
        DenseRole::Router,
        DenseRole::OutputHead,
    ] {
        let plan = sm12.plan_dense(DensePlanRequest { role, ..request })?;
        assert_eq!(plan.execution(), DenseExecution::Vector);
        assert_eq!(plan.source(), PlanSource::Heuristic);
    }
    assert_eq!(
        sm12.plan_dense(DensePlanRequest { role: DenseRole::DenseGateUp, ..request })?
            .execution(),
        DenseExecution::Matrix
    );
    assert_eq!(
        sm12.plan_dense(DensePlanRequest { tokens: 2, ..request })?.execution(),
        DenseExecution::Matrix
    );
    assert_eq!(planner(11)?.plan_dense(request)?.execution(), DenseExecution::Matrix);
    Ok(())
}

#[test]
fn explicit_vendor_policy_accepts_generic_aligned_geometry() -> Result<()> {
    let planner = planner_with_policy(
        12,
        CudaPlanningPolicy {
            numerical: CudaNumericalPolicy::Throughput,
            admission: CudaKernelAdmission::Experimental,
            dense_vendor: CudaDenseVendorPolicy::Tuned,
            ..CudaPlanningPolicy::default()
        },
    )?;
    let request = DensePlanRequest {
        phase: ExecutionPhase::Decode,
        role: DenseRole::DenseDown,
        tokens: 1,
        input_features: 3_072,
        output_features: 4_096,
    };
    for candidate in [
        request,
        DensePlanRequest {
            phase: ExecutionPhase::Prefill,
            tokens: 384,
            ..request
        },
        DensePlanRequest {
            role: DenseRole::Router,
            output_features: 64,
            ..request
        },
    ] {
        let plan = planner.plan_dense(candidate)?;
        assert_eq!(plan.execution(), DenseExecution::CublasLt);
        assert_eq!(plan.source(), PlanSource::ExplicitPolicy);
    }
    assert_eq!(
        planner
            .plan_dense(DensePlanRequest { input_features: 3_073, ..request })?
            .execution(),
        DenseExecution::Matrix
    );
    Ok(())
}
