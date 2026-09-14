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
        let packed = (0..8 * n * k / 8)
            .map(|i| Ok(u32::try_from(i)?.wrapping_mul(0x91a5_372d).wrapping_add(0xfedc_ba98)))
            .collect::<Result<Vec<_>>>()?;
        let weight = Array::from_u32(&packed, &[8, i32::try_from(n)?, i32::try_from(k / 8)?])?;
        let scales = (0..8 * n * k / 32)
            .map(|i| Ok(123 + u32::try_from(i % 6)?))
            .collect::<Result<Vec<_>>>()?;
        let scales = Array::from_u32(&scales, &[8, i32::try_from(n)?, i32::try_from(k / 32)?])?
            .astype(Dtype::Uint8, &stream)?;
        let expected = stream.native().graph().gather_mxfp4_with_indices(
            input.native(),
            mirtal::MxFp4 {
                weight: weight.native(),
                scales: scales.native(),
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
            let differences =
                actual.iter().zip(&expected).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            assert_eq!(differences, 0, "K={k}, N={n}, columns={columns:?}");
        }
    }
    Ok(())
}
