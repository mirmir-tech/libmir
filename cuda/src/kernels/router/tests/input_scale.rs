//! Llama 4's input scaling: each row times its one routing weight, which
//! then becomes one.

use mircuda::bf16;

use super::Runtime;
use crate::{Result, kernels::RouterInputScale};

const ROWS: usize = 3;
const COLUMNS: usize = 300;

#[test]
fn scales_each_row_by_its_weight_and_leaves_unit_weights() -> Result<()> {
    let runtime = Runtime::new()?;
    let input = (0..ROWS * COLUMNS)
        .map(|index| bf16::from_f32(f32::from(u16::try_from(index % 97).unwrap_or(0)) / 16.0 - 3.0))
        .collect::<Vec<_>>();
    let weights = [0.25_f32, 0.75, 1.5].map(bf16::from_f32);
    let scale = RouterInputScale::compile(&runtime.compiler, COLUMNS)?;
    let device_input = runtime.copy(&input)?;
    let mut device_weights = runtime.copy(&weights)?;
    let mut output = runtime.pool.allocate::<bf16>(&runtime.stream, ROWS * COLUMNS)?;
    scale.execute(&runtime.stream, &device_input, &mut device_weights, &mut output)?;
    let (output, after) = (runtime.read(&output)?, runtime.read(&device_weights)?);
    assert!(after.iter().all(|weight| *weight == bf16::ONE));
    for (index, value) in output.iter().enumerate() {
        let expected = bf16::from_f32(input[index].to_f32() * weights[index / COLUMNS].to_f32());
        assert_eq!(*value, expected, "element {index}");
    }
    Ok(())
}
