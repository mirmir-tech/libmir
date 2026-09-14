use std::io::Write;

use super::Result;

pub(super) fn difference(label: &str, a: &[f32], b: &[f32]) -> Result<()> {
    assert_eq!(a.len(), b.len());
    assert!(a.iter().chain(b).all(|value| value.is_finite()));
    let differing = a.iter().zip(b).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    let max_abs = a.iter().zip(b).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
    let squared = a
        .iter()
        .zip(b)
        .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
        .sum::<f64>();
    let norm = a.iter().map(|value| f64::from(*value).powi(2)).sum::<f64>().sqrt();
    let mut record = serde_json::json!({"label": label, "elements": a.len(), "differing": differing,
        "max_abs": max_abs, "relative_l2": squared.sqrt()/norm.max(f64::MIN_POSITIVE)});
    if label.contains("logits") || label.contains("chunk31/") {
        let width = a.len() / 5;
        record["tokens_a"] =
            serde_json::json!(a.chunks_exact(width).map(argmax).collect::<Vec<_>>());
        record["tokens_b"] =
            serde_json::json!(b.chunks_exact(width).map(argmax).collect::<Vec<_>>());
    }
    writeln!(std::io::stderr().lock(), "prefill.parity: {record}")?;
    Ok(())
}

fn argmax(values: &[f32]) -> usize {
    crate::native::benchmark::argmax(values).unwrap_or(0)
}
