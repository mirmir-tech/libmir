use std::num::NonZeroUsize;

use super::*;

#[expect(clippy::float_cmp, reason = "compare discrete BF16 rounding outcomes")]
pub(super) fn compare(
    projection: &MxFp4Linear,
    input: &Array,
    label: &str,
    stream: &Stream,
) -> Result<()> {
    let shape = input.shape()?;
    let batch = usize::try_from(shape[0])?;
    let sequence = usize::try_from(shape[1])?;
    let width = usize::try_from(shape[2])?;
    let flat = input.reshape(&[1, shape[0] * shape[1], shape[2]], stream)?;
    let reference = reference::dot_samples(projection, &flat, stream)?;
    let plan = mirtal::MxFp4Matmul::new()?;
    let weights = mirtal::MxFp4 {
        weight: projection.weight.native(),
        scales: projection.scales.native(),
    };
    for parts in [1, 4, 8] {
        let partitions = NonZeroUsize::new(parts).ok_or(Error::ShapeOverflow)?;
        let compiled = compiled(stream, partitions, mirtal::MxFp4RowTile::Rows16)?;
        let wide = self::compiled(stream, partitions, mirtal::MxFp4RowTile::Rows32)?;
        let packed = Array::from_native(plan.matmul(
            flat.native(),
            weights,
            partitions,
            stream.native(),
        )?)?
        .to_vec_f32(stream)?;
        let [replayed] =
            compiled.call(stream.native(), [flat.native(), weights.weight, weights.scales])?;
        assert_eq!(
            Array::from_native(replayed)?.to_vec_f32(stream)?,
            packed,
            "compiled packed MXFP4"
        );
        let [tiled] =
            wide.call(stream.native(), [flat.native(), weights.weight, weights.scales])?;
        assert_eq!(Array::from_native(tiled)?.to_vec_f32(stream)?, packed, "BM32 packed MXFP4");
        let mut scalar = Vec::new();
        for row in 0..batch {
            let input = input.slice(&[row, 0, 0], &[row + 1, sequence, width], stream)?;
            let [replayed] =
                compiled.call(stream.native(), [input.native(), weights.weight, weights.scales])?;
            let replayed = Array::from_native(replayed)?.to_vec_f32(stream)?;
            let [tiled] =
                wide.call(stream.native(), [input.native(), weights.weight, weights.scales])?;
            assert_eq!(
                Array::from_native(tiled)?.to_vec_f32(stream)?,
                replayed,
                "BM32 scalar MXFP4"
            );
            let start = scalar.len();
            scalar.extend(
                Array::from_native(plan.matmul(
                    input.native(),
                    weights,
                    partitions,
                    stream.native(),
                )?)?
                .to_vec_f32(stream)?,
            );
            assert_eq!(&scalar[start..], replayed, "compiled scalar MXFP4");
        }
        assert_eq!(scalar, packed, "mirtal FP32 candidate row-count stability");
        let sampled = packed
            .chunks_exact(projection.output_features)
            .flat_map(|row| {
                reference::columns(projection.output_features)
                    .into_iter()
                    .map(|column| f64::from(row[column]))
            })
            .collect::<Vec<_>>();
        assert_eq!(sampled.len(), reference.len());
        assert!(packed.iter().all(|v| v.is_finite()));
        let differing = sampled
            .iter()
            .zip(&reference)
            .filter(|(a, b)| **a != reference::nearest_bf16(**b))
            .count();
        writeln!(
            std::io::stderr().lock(),
            "mxfp4.candidate: {}",
            serde_json::json!({"label":label,"sequence":sequence,"batch":batch,"partitions":parts,"samples":sampled.len(),"not_nearest_bf16":differing,"row_count_exact":true,"compiled_exact":true,"row_tile_32_exact":true})
        )?;
        assert_eq!(
            differing, 0,
            "candidate misses the nearest BF16 reference in the captured sample"
        );
    }
    Ok(())
}

pub(super) fn compiled(
    stream: &Stream,
    partitions: NonZeroUsize,
    tile: mirtal::MxFp4RowTile,
) -> Result<mirtal::Compiled<3, 1>> {
    let plan = mirtal::MxFp4Matmul::new()?.with_row_tile(tile);
    Ok(stream.native().compile(
        mirtal::CompileOptions::default(),
        move |graph, [input, weight, scales]| {
            Ok([plan.matmul_graph(
                graph,
                &input,
                mirtal::MxFp4 { weight: &weight, scales: &scales },
                partitions,
            )?])
        },
    )?)
}
