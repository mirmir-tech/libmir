use super::*;

#[test]
fn weight_only_preserves_prefill_and_tunes_decode_buckets() {
    let decode = MoePlanRequest::nvfp4(ExecutionPhase::Decode, 1, 128, 8, 2_048, 768);
    let batch = MoePlanRequest { tokens: 4, ..decode };
    let prefill = MoePlanRequest {
        phase: ExecutionPhase::Prefill,
        tokens: 128,
        ..decode
    };

    assert_eq!(
        candidate_executions(decode, false),
        [MoeExecution::HybridW4A4, MoeExecution::IndexedGrouped]
    );
    assert!(candidate_executions(batch, false).is_empty());
    assert_eq!(
        candidate_executions(prefill, false),
        [MoeExecution::Bucketed, MoeExecution::IndexedGrouped, MoeExecution::SelectedWeightOnly,]
    );
    assert_eq!(
        candidate_executions(decode, true),
        [
            MoeExecution::HybridW4A4,
            MoeExecution::IndexedGrouped,
            MoeExecution::SelectedWeightOnly,
            MoeExecution::SelectedWeightOnlyTensorCore,
            MoeExecution::SelectedWeightOnlyTiled2,
            MoeExecution::SelectedWeightOnlyTiled4,
            MoeExecution::SelectedWeightOnlyTiled8,
            MoeExecution::MarlinWeightOnlyN128K128,
            MoeExecution::MarlinWeightOnlyN128K64,
            MoeExecution::MarlinWeightOnlyN64K128,
        ]
    );
    assert_eq!(
        candidate_executions(batch, true),
        [
            MoeExecution::Bucketed,
            MoeExecution::IndexedGrouped,
            MoeExecution::SelectedWeightOnly,
            MoeExecution::SelectedWeightOnlyTensorCore,
            MoeExecution::SelectedWeightOnlyTiled2,
            MoeExecution::SelectedWeightOnlyTiled4,
            MoeExecution::SelectedWeightOnlyTiled8,
            MoeExecution::MarlinWeightOnlyN128K128,
            MoeExecution::MarlinWeightOnlyN128K64,
            MoeExecution::MarlinWeightOnlyN64K128,
        ]
    );
    assert_eq!(candidate_executions(prefill, true), candidate_executions(prefill, false));
}

#[test]
fn numerical_gate_accepts_bf16_rounding_only() {
    let reference = [1.0, -20.0, 0.0].map(bf16::from_f32);
    let close = [1.5, -20.5, 0.5].map(bf16::from_f32);
    let drift = [1.75, -20.0, 0.0].map(bf16::from_f32);

    assert!(equivalent(&reference, &close));
    assert!(!equivalent(&reference, &drift));
}
