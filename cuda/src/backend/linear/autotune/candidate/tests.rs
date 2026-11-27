use super::*;
use crate::ExecutionPhase;

#[test]
fn prefill_uses_vendor_then_planner_fallback() {
    assert_eq!(
        initial_executions(DenseExecution::Matrix, None, ExecutionPhase::Prefill),
        [DenseExecution::CublasLt, DenseExecution::Matrix]
    );
    assert_eq!(
        initial_executions(DenseExecution::Vector, None, ExecutionPhase::Decode),
        [DenseExecution::Vector]
    );
}

#[test]
fn cached_execution_precedes_planner_fallback() {
    assert_eq!(
        initial_executions(
            DenseExecution::Matrix,
            Some(DenseExecution::Vector),
            ExecutionPhase::Decode
        ),
        [DenseExecution::Vector, DenseExecution::Matrix]
    );
}

#[test]
fn decode_measures_both_vector_kernels_the_width_admits() {
    let request = DensePlanRequest {
        phase: ExecutionPhase::Decode,
        role: crate::DenseRole::Router,
        tokens: 1,
        input_features: 2_048,
        output_features: 128,
    };
    let vectors = [DenseExecution::Vector, DenseExecution::RowGroupVector];
    assert!(vectors.iter().all(|execution| candidates(request, 48).contains(execution)));
    let odd = candidates(DensePlanRequest { input_features: 2_050, ..request }, 48);
    assert!(!odd.contains(&DenseExecution::Vector));
    assert!(odd.contains(&DenseExecution::RowGroupVector));
    let prefill = candidates(DensePlanRequest { tokens: 2, ..request }, 48);
    assert!(vectors.iter().all(|execution| !prefill.contains(execution)));
}

#[test]
fn decode_measures_split_k_sized_to_the_device() {
    let split = |splits| {
        std::num::NonZeroU16::new(splits).map(|splits| DenseExecution::SplitKVector { splits })
    };
    let splits = |request, multiprocessors| {
        candidates(request, multiprocessors)
            .into_iter()
            .filter(|execution| matches!(execution, DenseExecution::SplitKVector { .. }))
            .map(Some)
            .collect::<Vec<_>>()
    };
    let router = DensePlanRequest {
        phase: ExecutionPhase::Decode,
        role: crate::DenseRole::Router,
        tokens: 1,
        input_features: 2_048,
        output_features: 32,
    };
    assert_eq!(splits(router, 48), [split(3), split(6)]);
    assert_eq!(splits(router, 96), [split(6), split(12)]);
    assert_eq!(
        splits(DensePlanRequest { output_features: 64, ..router }, 48),
        [split(2), split(4)]
    );
    // Routers that fill two waves with one block per row measure two slices.
    assert_eq!(splits(DensePlanRequest { output_features: 128, ..router }, 48), [split(2)]);
    assert!(splits(DensePlanRequest { input_features: 2_050, ..router }, 48).is_empty());
    assert!(splits(DensePlanRequest { tokens: 2, ..router }, 48).is_empty());
}
