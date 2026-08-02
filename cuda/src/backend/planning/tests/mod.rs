use super::*;

mod attention;
mod output;
mod vendor;
mod weights;

fn planner_with_policy(major: u32, policy: CudaPlanningPolicy) -> Result<CudaExecutionPlanner> {
    Ok(CudaExecutionPlanner::new(
        CudaHardwareProfile::new(
            (major, 0),
            48,
            128 * 1_024 * 1_024 * 1_024,
            CudaMemoryArchitecture::Unified,
        )?,
        policy,
    ))
}

fn planner(major: u32) -> Result<CudaExecutionPlanner> {
    planner_with_policy(major, CudaPlanningPolicy::default())
}

#[test]
fn sm12_uses_validated_vectors_for_decode_output_and_attention() -> Result<()> {
    let output = DensePlanRequest {
        phase: ExecutionPhase::Decode,
        role: DenseRole::OutputHead,
        tokens: 1,
        input_features: 2_816,
        output_features: 262_144,
    };
    assert_eq!(planner(12)?.plan_dense(output)?.execution(), DenseExecution::Vector);
    let attention =
        planner(12)?.plan_dense(DensePlanRequest { role: DenseRole::AttentionQkv, ..output })?;
    assert_eq!(attention.execution(), DenseExecution::Vector);
    assert_eq!(attention.source(), PlanSource::Heuristic);
    assert_eq!(planner(11)?.plan_dense(output)?.execution(), DenseExecution::Matrix);
    Ok(())
}

#[test]
fn explicit_policy_admits_generic_decode_vectors() -> Result<()> {
    let planner = planner_with_policy(
        12,
        CudaPlanningPolicy {
            numerical: CudaNumericalPolicy::Throughput,
            admission: CudaKernelAdmission::Experimental,
            dense_vectors: CudaDenseVectorPolicy::Tuned,
            ..CudaPlanningPolicy::default()
        },
    )?;
    let qkv = DensePlanRequest {
        phase: ExecutionPhase::Decode,
        role: DenseRole::AttentionQkv,
        tokens: 1,
        input_features: 2_816,
        output_features: 8_192,
    };
    let plan = planner.plan_dense(qkv)?;
    assert_eq!(plan.execution(), DenseExecution::Vector);
    assert_eq!(plan.source(), PlanSource::ExplicitPolicy);
    for (role, input_features, output_features) in
        [(DenseRole::AttentionQkv, 2_880, 5_120), (DenseRole::AttentionOutput, 4_096, 2_880)]
    {
        assert_eq!(
            planner
                .plan_dense(DensePlanRequest {
                    role,
                    input_features,
                    output_features,
                    ..qkv
                })?
                .execution(),
            DenseExecution::Vector
        );
    }
    for request in [
        DensePlanRequest { phase: ExecutionPhase::Prefill, ..qkv },
        DensePlanRequest { tokens: 2, ..qkv },
    ] {
        assert_eq!(planner.plan_dense(request)?.execution(), DenseExecution::Matrix);
    }
    assert_eq!(
        planner
            .plan_dense(DensePlanRequest {
                role: DenseRole::DenseDown,
                input_features: 4_096,
                ..qkv
            })?
            .execution(),
        DenseExecution::Vector
    );
    Ok(())
}

#[test]
fn phase_selects_current_nvfp4_strategy() -> Result<()> {
    let request = MoePlanRequest {
        phase: ExecutionPhase::Decode,
        quantization: MoeQuantization::NvFp4,
        tokens: 1,
        experts: 128,
        top_k: 4,
        hidden_features: 2_816,
        intermediate_features: 1_408,
    };
    assert_eq!(planner(12)?.plan_moe(request)?.execution(), MoeExecution::HybridW4A4);
    assert_eq!(planner(11)?.plan_moe(request)?.execution(), MoeExecution::IndexedGrouped);
    assert_eq!(
        planner(12)?.plan_moe(MoePlanRequest { tokens: 16, ..request })?.execution(),
        MoeExecution::Bucketed
    );
    assert_eq!(
        planner(12)?.plan_moe(MoePlanRequest { tokens: 15, ..request })?.execution(),
        MoeExecution::IndexedGrouped
    );
    let forced_indexed = planner_with_policy(
        12,
        CudaPlanningPolicy {
            moe_batch: CudaMoeBatchPolicy::W4A4,
            ..CudaPlanningPolicy::default()
        },
    )?;
    assert_eq!(
        forced_indexed.plan_moe(MoePlanRequest { tokens: 16, ..request })?.execution(),
        MoeExecution::IndexedGrouped
    );
    assert_eq!(forced_indexed.plan_moe(request)?.execution(), MoeExecution::IndexedGrouped);
    let direct = planner_with_policy(
        12,
        CudaPlanningPolicy {
            moe_batch: CudaMoeBatchPolicy::W4A4Direct,
            ..CudaPlanningPolicy::default()
        },
    )?;
    assert_eq!(direct.plan_moe(request)?.execution(), MoeExecution::DirectW4A4);
    assert_eq!(
        direct.plan_moe(MoePlanRequest { tokens: 6, ..request })?.execution(),
        MoeExecution::IndexedGrouped
    );
    let hybrid = planner_with_policy(
        12,
        CudaPlanningPolicy {
            moe_batch: CudaMoeBatchPolicy::W4A4Hybrid,
            ..CudaPlanningPolicy::default()
        },
    )?;
    assert_eq!(hybrid.plan_moe(request)?.execution(), MoeExecution::HybridW4A4);
    let fused = planner_with_policy(
        12,
        CudaPlanningPolicy {
            numerical: CudaNumericalPolicy::Throughput,
            admission: CudaKernelAdmission::Experimental,
            moe_fusion: CudaMoeFusionPolicy::Tuned,
            ..CudaPlanningPolicy::default()
        },
    )?;
    assert_eq!(fused.plan_moe(request)?.execution(), MoeExecution::FusedIndexedGrouped);
    let bucketed = planner_with_policy(
        12,
        CudaPlanningPolicy {
            moe_batch: CudaMoeBatchPolicy::W4A4Bucketed,
            ..CudaPlanningPolicy::default()
        },
    )?;
    assert_eq!(
        bucketed.plan_moe(MoePlanRequest { tokens: 8, ..request })?.execution(),
        MoeExecution::Bucketed
    );
    assert_eq!(bucketed.plan_moe(request)?.execution(), MoeExecution::IndexedGrouped);
    assert_eq!(
        planner(12)?
            .plan_moe(MoePlanRequest {
                phase: ExecutionPhase::Prefill,
                tokens: 256,
                ..request
            })?
            .execution(),
        MoeExecution::Bucketed
    );
    Ok(())
}

#[test]
fn explicit_w4a16_covers_prefill_and_every_decode_depth() -> Result<()> {
    let planner = planner_with_policy(
        12,
        CudaPlanningPolicy {
            moe_batch: CudaMoeBatchPolicy::W4A16,
            ..CudaPlanningPolicy::default()
        },
    )?;
    let request = MoePlanRequest {
        phase: ExecutionPhase::Decode,
        quantization: MoeQuantization::NvFp4,
        tokens: 1,
        experts: 128,
        top_k: 4,
        hidden_features: 2_816,
        intermediate_features: 1_408,
    };
    for tokens in [1, 8] {
        assert_eq!(
            planner.plan_moe(MoePlanRequest { tokens, ..request })?.execution(),
            MoeExecution::SelectedWeightOnly
        );
    }
    assert_eq!(
        planner
            .plan_moe(MoePlanRequest {
                phase: ExecutionPhase::Prefill,
                tokens: 128,
                ..request
            })?
            .execution(),
        MoeExecution::SelectedWeightOnly
    );
    Ok(())
}
