pub(super) fn difference(a: &[f32], b: &[f32]) -> serde_json::Value {
    assert_eq!(a.len(), b.len());
    assert!(a.iter().chain(b).all(|value| value.is_finite()));
    let squared = a
        .iter()
        .zip(b)
        .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
        .sum::<f64>();
    let norm = a.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>().sqrt();
    serde_json::json!({
        "elements":a.len(),
        "differing":a.iter().zip(b).filter(|(a,b)| a.to_bits()!=b.to_bits()).count(),
        "max_abs":a.iter().zip(b).map(|(a,b)| (a-b).abs()).fold(0.0,f32::max),
        "relative_l2":squared.sqrt()/norm.max(f64::MIN_POSITIVE)
    })
}

pub(super) fn logits(a: &[f32], b: &[f32]) -> serde_json::Value {
    let mut result = difference(a, b);
    let log_partition = |values: &[f32]| {
        let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        f64::from(max)
            + values.iter().map(|v| (f64::from(*v) - f64::from(max)).exp()).sum::<f64>().ln()
    };
    let az = log_partition(a);
    let bz = log_partition(b);
    let kl = a
        .iter()
        .zip(b)
        .map(|(a, b)| {
            let logp = f64::from(*a) - az;
            logp.exp() * (logp - (f64::from(*b) - bz))
        })
        .sum::<f64>();
    result["kl_scalar_to_packed"] = serde_json::json!(kl);
    result["scalar_token"] = serde_json::json!(argmax(a));
    result["packed_token"] = serde_json::json!(argmax(b));
    result
}

pub(super) fn argmax(values: &[f32]) -> usize {
    crate::native::benchmark::argmax(values).unwrap_or(0)
}
