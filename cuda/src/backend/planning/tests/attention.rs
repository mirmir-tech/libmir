use super::*;

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
            partition_tokens: 256,
            threshold_tokens: 65
        }
    );
    assert_eq!(tuned.source(), PlanSource::Heuristic);
    let tensor = AttentionPlanRequest {
        head_dim: 64,
        value_head_dim: 64,
        ..request
    };
    assert_eq!(
        planner(12)?.plan_attention(tensor)?.execution(),
        AttentionExecution::SplitKv {
            partition_tokens: 384,
            threshold_tokens: 65
        }
    );
    let mha = AttentionPlanRequest { kv_heads: request.query_heads, ..request };
    assert_eq!(
        planner(12)?.plan_attention(mha)?.execution(),
        AttentionExecution::SplitKv {
            partition_tokens: 64,
            threshold_tokens: 65
        }
    );
    assert_eq!(planner(11)?.plan_attention(request)?.execution(), AttentionExecution::Direct);
    let wide = AttentionPlanRequest {
        head_dim: 256,
        value_head_dim: 256,
        ..request
    };
    assert_eq!(
        planner(12)?.plan_attention(wide)?.execution(),
        AttentionExecution::SplitKv {
            partition_tokens: 256,
            threshold_tokens: 512
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
