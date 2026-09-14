use super::*;

#[test]
fn packed_fused_decode_preserves_normalization_gates_and_continuation() -> Result<()> {
    let stream = crate::engine::Stream::new_gpu()?;
    let native = stream.native();
    let packed = super::super::decode::packed_decode()?;
    for dtype in [DType::Bfloat16, DType::Float16, DType::Float32] {
        for normalize in [false, true] {
            let case = Case {
                batch: 3,
                steps: 1,
                key_heads: 2,
                value_heads: 4,
                key_dim: 128,
                value_dim: 128,
                dtype,
            };
            let [query, key, value, _, _, state] = case.inputs(native)?;
            let alpha = data(native, [3, 1, 4], dtype, 7, 1.0)?;
            let beta = data(native, [3, 1, 4], dtype, 8, 1.0)?;
            let log = data(native, [4], DType::Float32, 9, 0.5)?;
            let bias = data(native, [4], DType::Float32, 10, 0.5)?;
            let mut baseline = state.clone();
            let mut candidate = state;
            for step in 0..128 {
                let expected = stream.gated_delta_decode(
                    [&query, &key, &value, &alpha, &beta, &log, &bias, &baseline],
                    normalize,
                )?;
                let inputs = [&query, &key, &value, &alpha, &beta, &log, &bias, &candidate];
                let actual = packed.dispatch(
                    native,
                    inputs,
                    &[
                        OutputSpec::new(value.shape()?, dtype),
                        OutputSpec::new(candidate.shape()?, DType::Float32),
                    ],
                    &super::super::decode::dispatch(inputs, normalize)?,
                )?;
                exact(
                    native,
                    &expected,
                    &actual,
                    &format!("fused {dtype:?} normalize={normalize} step={step}"),
                )?;
                native.synchronize()?;
                baseline = expected[1].clone();
                candidate = actual[1].clone();
                native.detach_graph(&baseline)?;
                native.detach_graph(&candidate)?;
            }
        }
    }
    Ok(())
}
