use super::*;

mod output;

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
fn attention_policy_selects_tuned_sm12_split_kv() -> Result<()> {
    let request = AttentionPlanRequest {
        max_context_tokens: 4_096,
        query_heads: 32,
        kv_heads: 8,
        head_dim: 128,
        value_head_dim: 128,
    };
    let tuned = planner(12)?.plan_attention(request)?;
    assert_eq!(
        tuned.execution(),
        AttentionExecution::SplitKv {
            partition_tokens: 64,
            threshold_tokens: 65
        }
    );
    assert_eq!(tuned.source(), PlanSource::Tuned);
    assert_eq!(planner(11)?.plan_attention(request)?.execution(), AttentionExecution::Direct);
    let wide = AttentionPlanRequest {
        head_dim: 256,
        value_head_dim: 256,
        ..request
    };
    assert_eq!(
        planner(12)?.plan_attention(wide)?.execution(),
        AttentionExecution::SplitKv {
            partition_tokens: 64,
            threshold_tokens: 128
        }
    );
    let direct = planner_with_policy(
        12,
        CudaPlanningPolicy {
            attention: CudaAttentionPolicy::Direct,
            ..CudaPlanningPolicy::default()
        },
    )?;
    assert_eq!(direct.plan_attention(request)?.execution(), AttentionExecution::Direct);
    let explicit = planner_with_policy(
        11,
        CudaPlanningPolicy {
            attention: CudaAttentionPolicy::SplitKv {
                partition_tokens: 128,
                threshold_tokens: 384,
            },
            ..CudaPlanningPolicy::default()
        },
    )?;
    assert_eq!(
        explicit.plan_attention(request)?.execution(),
        AttentionExecution::SplitKv {
            partition_tokens: 128,
            threshold_tokens: 384
        }
    );
    Ok(())
}

#[test]
fn sm12_uses_validated_vector_only_for_decode_output() -> Result<()> {
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
    assert_eq!(attention.execution(), DenseExecution::Matrix);
    assert_eq!(attention.source(), PlanSource::Fallback);
    assert_eq!(planner(11)?.plan_dense(output)?.execution(), DenseExecution::Matrix);
    Ok(())
}

#[test]
fn explicit_policy_admits_only_tuned_decode_vectors() -> Result<()> {
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
    for request in [
        DensePlanRequest { phase: ExecutionPhase::Prefill, ..qkv },
        DensePlanRequest {
            role: DenseRole::DenseDown,
            output_features: 2_816,
            ..qkv
        },
        DensePlanRequest { input_features: 4_096, ..qkv },
    ] {
        assert_eq!(planner.plan_dense(request)?.execution(), DenseExecution::Matrix);
    }
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
    let weight_only = planner_with_policy(
        12,
        CudaPlanningPolicy {
            moe_batch: CudaMoeBatchPolicy::W4A16,
            ..CudaPlanningPolicy::default()
        },
    )?;
    assert_eq!(
        weight_only.plan_moe(MoePlanRequest { tokens: 8, ..request })?.execution(),
        MoeExecution::SelectedWeightOnly
    );
    assert_eq!(weight_only.plan_moe(request)?.execution(), MoeExecution::IndexedGrouped);
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
