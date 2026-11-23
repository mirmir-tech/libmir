//! Llama 4's query temperature: queries scale by
//! `1 + beta · ln(1 + floor((p + offset) / window))`, keys and values do not.

use mircuda::bf16;
use models::semantic::QueryTemperature;

use super::{Runtime, batched, fused, test_values};
use crate::{
    Result,
    kernels::{QkvNormalization, QkvPostprocessSpec, QueryTemperatureScale},
};

const TEMPERATURE: QueryTemperature = QueryTemperature { beta: 0.5, window: 4, offset: 1 };

fn spec() -> Result<QkvPostprocessSpec> {
    Ok(QkvPostprocessSpec {
        tokens: 2,
        query_heads: 1,
        kv_heads: 1,
        head_dim: 8,
        value_head_dim: 8,
        rotary: None,
        epsilon: 1.0e-6,
        normalization: QkvNormalization::NONE,
        temperature: Some(QueryTemperatureScale::new(TEMPERATURE)?),
    })
}

#[test]
fn queries_temper_by_position_and_keys_do_not() -> Result<()> {
    let runtime = Runtime::new()?;
    let (packed, weight) = (test_values(48, 0)?, test_values(8, 0)?);
    // Positions 6 and 7 cross one and two windows with the one-based offset.
    let fused = fused(&runtime, spec()?, &packed, &weight, &weight, 6)?;
    // The batched helper places its rows at positions 3 and 4.
    let batched = batched(&runtime, spec()?, &packed, &weight, &weight)?;
    for ((query, key, value), positions) in [(fused, [6, 7]), (batched, [3, 4])] {
        for (token, position) in positions.into_iter().enumerate() {
            let row = &packed[token * 24..token * 24 + 24];
            let at = token * 8..token * 8 + 8;
            #[allow(clippy::cast_possible_truncation)]
            let factor = TEMPERATURE.factor(position) as f32;
            for (actual, input) in query[at.clone()].iter().zip(&row[..8]) {
                assert_eq!(*actual, bf16::from_f32(input.to_f32() * factor), "at {position}");
            }
            assert_eq!(&key[at.clone()], &row[8..16]);
            assert_eq!(&value[at], &row[16..]);
        }
    }
    Ok(())
}
