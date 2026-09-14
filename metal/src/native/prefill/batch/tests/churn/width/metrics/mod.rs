mod tests;

use std::io::Write;

use super::{Frame, Result};

pub(super) fn assert_identical(a: &Frame, b: &Frame, message: &str) {
    assert_eq!(a.logits.len(), b.logits.len(), "{message}");
    for (a, b) in a.logits.iter().zip(&b.logits) {
        assert_eq!(a.len(), b.len(), "{message}");
        assert!(a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()), "{message}");
    }
}

pub(super) fn argmax(values: &[f32]) -> Result<u32> {
    let index = crate::native::benchmark::argmax(values)
        .ok_or(crate::native::error::Error::NoPendingDecode)?;
    Ok(u32::try_from(index)?)
}

pub(super) fn report(step: usize, scalar: &Frame, packed: &Frame) -> Result<()> {
    let stages = packed.stages.len();
    assert_eq!(scalar.stages.len(), stages * scalar.logits.len());
    for (index, batch) in packed.stages.iter().enumerate() {
        let single = &scalar.stages[index];
        assert_eq!((single.layer, single.stage), (batch.layer, batch.stage));
        let width = single.values.len();
        let row = &batch.values[..width];
        let difference = difference(&single.values, row);
        writeln!(
            std::io::stderr().lock(),
            "width.stage: {}",
            serde_json::json!({
                "step":step,"layer":single.layer,"stage":single.stage,"row":0,"difference":difference
            })
        )?;
    }
    for (row, (single, batch)) in scalar.logits.iter().zip(&packed.logits).enumerate() {
        writeln!(
            std::io::stderr().lock(),
            "width.logits: {}",
            serde_json::json!({
                "step":step,"row":row,"difference":difference(single,batch),
                "scalar_top":top(single),"packed_top":top(batch)
            })
        )?;
    }
    Ok(())
}

fn difference(a: &[f32], b: &[f32]) -> serde_json::Value {
    assert_eq!(a.len(), b.len());
    assert!(a.iter().chain(b).all(|v| v.is_finite()));
    let changed = a.iter().zip(b).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    let max_abs = a.iter().zip(b).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
    serde_json::json!({"values":a.len(),"changed":changed,"max_abs":max_abs})
}

fn top(values: &[f32]) -> Vec<(usize, f32)> {
    let mut values = values.iter().copied().enumerate().collect::<Vec<_>>();
    values.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    values.truncate(5);
    values
}
