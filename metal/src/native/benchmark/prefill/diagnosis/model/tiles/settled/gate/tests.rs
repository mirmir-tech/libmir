use super::*;

fn pair(a: f64, b: f64) -> [Timing; 2] {
    [Timing { prefill_ms: a, decode_ms: 10.0 }, Timing { prefill_ms: b, decode_ms: 9.0 }]
}

#[test]
fn requires_three_rounds_for_each_plan_and_both_phases() {
    assert_eq!(warmup_status(&[]), WarmupStatus::Warming);
    assert_eq!(warmup_status(&[pair(100.0, 90.0); 2]), WarmupStatus::Warming);
    let ready = [pair(100.0, 90.0), pair(102.0, 92.0), pair(101.0, 91.0)];
    assert_eq!(warmup_status(&ready), WarmupStatus::Ready);
    let mut decode_drift = ready;
    decode_drift[2][1].decode_ms = 12.0;
    assert_eq!(warmup_status(&decode_drift), WarmupStatus::Warming);
    let mut prefill_drift = ready;
    prefill_drift[2][0].prefill_ms = 120.0;
    assert_eq!(warmup_status(&prefill_drift), WarmupStatus::Warming);
}

#[test]
fn accepts_a_plateau_after_cold_start_but_bounds_persistent_drift() {
    let plateau = [pair(200.0, 180.0), pair(100.0, 90.0), pair(101.0, 91.0), pair(100.0, 90.0)];
    assert_eq!(warmup_status(&plateau), WarmupStatus::Ready);
    let drifting = [
        pair(160.0, 150.0),
        pair(150.0, 140.0),
        pair(140.0, 130.0),
        pair(130.0, 120.0),
        pair(120.0, 110.0),
        pair(110.0, 100.0),
    ];
    assert_eq!(warmup_status(&drifting), WarmupStatus::Exhausted);
    let mut final_plateau = drifting;
    final_plateau[3..].fill(pair(100.0, 90.0));
    assert_eq!(warmup_status(&final_plateau), WarmupStatus::Ready);
}

#[test]
fn rejects_nonfinite_and_nonpositive_measurements() {
    for bad in [f64::NAN, f64::INFINITY, 0.0, -1.0] {
        let mut rounds = [pair(100.0, 90.0); MAX_ROUNDS];
        rounds[MAX_ROUNDS - 1][0].prefill_ms = bad;
        assert_eq!(warmup_status(&rounds), WarmupStatus::Exhausted);
    }
}
