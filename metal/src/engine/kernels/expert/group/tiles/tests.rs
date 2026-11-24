use super::*;

#[test]
fn worklist_covers_compact_routes_once_with_empty_experts_and_tail_tiles() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let plan = TilePlan::new()?;
    for counts in [[0, 1, 0, 17, 32, 0, 3, 0], [0, 0, 0, 0, 0, 0, 0, 65], [16; 8]] {
        let ids = counts
            .iter()
            .zip(0_u32..)
            .flat_map(|(&count, id)| std::iter::repeat_n(id, count))
            .collect::<Vec<_>>();
        let indices = Array::from_u32(&ids, &[i32::try_from(ids.len())?])?;
        let tiles = plan.prepare(&indices, counts.len(), &stream)?;
        let descriptors = Array::from_native(tiles.descriptors)?.to_vec_u32(&stream)?;
        let mut covered = vec![0; ids.len()];
        let mut active = 0;
        for tile in descriptors.as_chunks::<3>().0 {
            let (first, end, expert) =
                (usize::try_from(tile[0])?, usize::try_from(tile[1])?, tile[2]);
            if first == end {
                assert_eq!(*tile, [0, 0, 0]);
                continue;
            }
            active += 1;
            assert!(end <= ids.len() && end - first <= 16);
            for row in first..end {
                assert_eq!(ids[row], expert);
                covered[row] += 1;
            }
        }
        assert!(covered.into_iter().all(|count| count == 1));
        assert_eq!(active, counts.into_iter().map(|count| count.div_ceil(16)).sum::<usize>());
    }
    Ok(())
}

#[test]
fn tile_mxfp4_matches_grouped_mlx_with_signed_weights_and_partial_groups() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let plan = TilePlan::new()?;
    for (k, n) in [(32, 32), (64, 96), (512, 64), (2048, 512), (512, 2048)] {
        let ids = [0_u32, 1, 4, 7]
            .into_iter()
            .zip([17, 1, 31, 16])
            .flat_map(|(id, count)| std::iter::repeat_n(id, count))
            .collect::<Vec<_>>();
        let routes = ids.len();
        let indices = Array::from_u32(&ids, &[i32::try_from(routes)?])?;
        let tiles = plan.prepare(&indices, 8, &stream)?;
        let input = (0..routes * k)
            .map(|i| Ok((f32::from(u16::try_from(i * 17 % 233)?) - 116.0) / 71.0))
            .collect::<Result<Vec<_>>>()?;
        let input = Array::from_f32(&input, &[i32::try_from(routes)?, 1, i32::try_from(k)?])?
            .astype(Dtype::Bfloat16, &stream)?;
        let values = input.to_vec_f32(&stream)?;
        // MLX 0.32.4's sorted gather reads one scale group past an expert at
        // K=32, so its operands carry a ninth, never-routed expert with valid
        // scales; the tile kernel reads the exact eight.
        let packed = (0..9 * n * k / 8)
            .map(|i| Ok(u32::try_from(i)?.wrapping_mul(0x91a5_372d).wrapping_add(0xfedc_ba98)))
            .collect::<Result<Vec<_>>>()?;
        let scale_bytes = (0..9 * n * k / 32)
            .map(|i| Ok(123 + u32::try_from(i % 6)?))
            .collect::<Result<Vec<_>>>()?;
        let experts = |count: i32| -> Result<(Array, Array)> {
            let experts = usize::try_from(count)?;
            let (n, k) = (i32::try_from(n)?, i32::try_from(k)?);
            let weight =
                Array::from_u32(&packed[..experts * packed.len() / 9], &[count, n, k / 8])?;
            let scales = Array::from_u32(
                &scale_bytes[..experts * scale_bytes.len() / 9],
                &[count, n, k / 32],
            )?
            .astype(Dtype::Uint8, &stream)?;
            Ok((weight, scales))
        };
        let (weight, scales) = experts(8)?;
        let (padded_weight, padded_scales) = experts(9)?;
        let expected = stream.native().graph().gather_mxfp4_with_indices(
            input.native(),
            mirtal::MxFp4 {
                weight: padded_weight.native(),
                scales: padded_scales.native(),
            },
            mirtal::GatherIndices { lhs: None, rhs: Some(indices.native()) },
            mirtal::GatherQmmOptions { transpose: true, sorted_indices: true },
        )?;
        let expected = Array::from_native(expected)?.to_vec_f32(&stream)?;
        for columns in [ColumnTile::Width32, ColumnTile::Width64] {
            let plan = TilePlan::with_columns(columns)?;
            let output = plan.project(&input, &weight, &scales, &tiles, &stream);
            if !n.is_multiple_of(columns.columns()) {
                assert!(output.is_err());
                continue;
            }
            let actual = output?.to_vec_f32(&stream)?;
            let reference = Reference {
                input: &values,
                packed: &packed,
                ids: &ids,
                k,
                n,
            };
            // The reference rounds to MLX's output within half a bf16 ulp.
            for index in (0..expected.len()).step_by(97) {
                let value = f64::from(expected[index]);
                let exact = reference.output(index / n, index % n);
                assert!((value - exact).abs() <= value.abs() / 256.0 + 1.0e-30, "index={index}");
            }
            for (index, (a, b)) in actual.iter().zip(&expected).enumerate() {
                if a.to_bits() == b.to_bits() {
                    continue;
                }
                // f32 accumulation order may round a bf16 tie differently;
                // the tile kernel must then be at least as close as MLX.
                let exact = reference.output(index / n, index % n);
                assert!(
                    one_ulp_apart(*a, *b)
                        && (f64::from(*a) - exact).abs() <= (f64::from(*b) - exact).abs(),
                    "K={k}, N={n}, columns={columns:?}, index={index}: {a} vs MLX {b}, exact {exact}"
                );
            }
        }
    }
    Ok(())
}

/// The float64 output of the test's MXFP4 experts: E2M1 nibbles from the low
/// bits, scaled per 32 inputs by `2^(scale - 127)` with scales `123 + i % 6`.
struct Reference<'a> {
    input: &'a [f32],
    packed: &'a [u32],
    ids: &'a [u32],
    k: usize,
    n: usize,
}

impl Reference<'_> {
    fn output(&self, route: usize, column: usize) -> f64 {
        const E2M1: [f64; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        let row = self.ids[route] as usize * self.n + column;
        (0..self.k)
            .map(|index| {
                let nibble = (self.packed[row * self.k / 8 + index / 8] >> (4 * (index % 8))) & 0xf;
                let magnitude = E2M1[(nibble & 7) as usize];
                let weight = if nibble & 8 == 0 {
                    magnitude
                } else {
                    -magnitude
                };
                let group = row * self.k / 32 + index / 32;
                let exponent = 123 + i32::try_from(group % 6).unwrap_or(0) - 127;
                f64::from(self.input[route * self.k + index]) * weight * 2.0_f64.powi(exponent)
            })
            .sum()
    }
}

/// Whether two bf16 values stored as f32 are adjacent.
fn one_ulp_apart(a: f32, b: f32) -> bool {
    ((a.to_bits() >> 16).abs_diff(b.to_bits() >> 16)) == 1
}
