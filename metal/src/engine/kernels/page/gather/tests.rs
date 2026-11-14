use super::*;
use crate::engine::Array;

#[test]
fn gathers_fragmented_independent_arenas_in_logical_order() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let kernel = Gather::new()?;
    for dtype in [mirtal::DType::Float32, mirtal::DType::Float16, mirtal::DType::Bfloat16] {
        for (heads, dim) in [(2, 256), (8, 64)] {
            for tokens in [1, 16, 17, 47] {
                for (width, ids) in [1, 3, 12]
                    .into_iter()
                    .flat_map(|width| [[0_u32, 1, 2], [4, 1, 4]].map(|ids| (width, ids)))
                {
                    let mut arenas = Vec::new();
                    let mut tables = Vec::new();
                    for row in 0..width {
                        let capacity = 5 + row;
                        let input = (0..heads * capacity * 16 * dim)
                            .map(|i| {
                                u8::try_from((i * 13 + row * 7) % 251)
                                    .map(|v| f32::from(v) / 8.0 - 15.0)
                            })
                            .collect::<std::result::Result<Vec<_>, _>>()?;
                        let keys = mirtal::Array::from_slice(&input, [heads, capacity, 16, dim])?;
                        let values = mirtal::Array::from_slice(
                            &input.iter().map(|v| -*v).collect::<Vec<_>>(),
                            [heads, capacity, 16, dim],
                        )?;
                        arenas.push([
                            stream.native().graph().astype(&keys, dtype)?,
                            stream.native().graph().astype(&values, dtype)?,
                        ]);
                        tables.push(mirtal::Array::from_slice(&ids, [3])?);
                    }
                    let sources = arenas
                        .iter()
                        .zip(&tables)
                        .map(|(a, table)| Source { keys: &a[0], values: &a[1], table })
                        .collect::<Vec<_>>();
                    let output = kernel.execute(&sources, tokens, &stream)?;
                    for kind in 0..2 {
                        let actual =
                            Array::from_native(output[kind].clone())?.to_vec_f32(&stream)?;
                        let mut expected = Vec::new();
                        let mut native = Vec::new();
                        for (row, arena) in arenas.iter().enumerate() {
                            let values =
                                Array::from_native(arena[kind].clone())?.to_vec_f32(&stream)?;
                            for head in 0..heads {
                                for token in 0..tokens {
                                    for column in 0..dim {
                                        expected.push(
                                            values[((head * (5 + row) + ids[token / 16] as usize)
                                                * 16
                                                + token % 16)
                                                * dim
                                                + column],
                                        );
                                    }
                                }
                            }
                            let graph = stream.native().graph();
                            let pages = tokens.div_ceil(16);
                            let table = graph.slice(&tables[row], &[0], &[pages])?;
                            let taken = graph.take(&arena[kind], &table, 1)?;
                            let reshaped = graph.reshape(
                                &taken,
                                &mirtal::Shape::new([1, heads, pages * 16, dim])?,
                            )?;
                            native.push(graph.slice(
                                &reshaped,
                                &[0, 0, 0, 0],
                                &[1, heads, tokens, dim],
                            )?);
                        }
                        assert_eq!(bits(&actual), bits(&expected));
                        let joined = stream
                            .native()
                            .graph()
                            .concatenate(&native.iter().collect::<Vec<_>>(), 0)?;
                        assert_eq!(
                            bits(&actual),
                            bits(&Array::from_native(joined)?.to_vec_f32(&stream)?)
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
fn rejects_invalid_gather_contracts() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let kernel = Gather::new()?;
    let keys = mirtal::Array::from_slice(&[1.0_f32; 16], [1, 1, 4, 4])?;
    let table = mirtal::Array::from_slice(&[0_u32], [1])?;
    let source = || Source {
        keys: &keys,
        values: &keys,
        table: &table,
    };
    assert!(kernel.execute(&[], 1, &stream).is_err());
    assert!(kernel.execute(&[source()], 0, &stream).is_err());
    assert!(kernel.execute(&[source()], 5, &stream).is_err());
    assert!(
        kernel
            .execute(&(0..13).map(|_| source()).collect::<Vec<_>>(), 1, &stream)
            .is_err()
    );
    let wrong = mirtal::Array::from_slice(&[1.0_f32], [1])?;
    assert!(kernel.execute(&[Source { values: &wrong, ..source() }], 1, &stream).is_err());
    assert!(kernel.execute(&[Source { table: &wrong, ..source() }], 1, &stream).is_err());
    Ok(())
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn gathers_partial_vector_and_non_power_of_two_page() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let kernel = Gather::new()?;
    let input = (0_u8..63).map(f32::from).collect::<Vec<_>>();
    let keys = mirtal::Array::from_slice(&input, [1, 3, 3, 7])?;
    let table = mirtal::Array::from_slice(&[2_u32, 0], [2])?;
    let expected = [&input[42..63], &input[..7]].concat();
    for dtype in [mirtal::DType::Float32, mirtal::DType::Float16, mirtal::DType::Bfloat16] {
        let keys = stream.native().graph().astype(&keys, dtype)?;
        let outputs = kernel.execute(
            &[Source {
                keys: &keys,
                values: &keys,
                table: &table,
            }],
            4,
            &stream,
        )?;
        for output in outputs {
            assert_eq!(bits(&Array::from_native(output)?.to_vec_f32(&stream)?), bits(&expected));
        }
    }
    Ok(())
}
