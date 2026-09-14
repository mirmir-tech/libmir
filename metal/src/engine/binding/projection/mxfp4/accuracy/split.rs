use super::*;

#[expect(clippy::float_cmp, reason = "compare discrete BF16 results of the reduction hypotheses")]
pub(super) fn compare(
    projection: &MxFp4Linear,
    input: &Array,
    label: &str,
    sequence: i32,
    batch: i32,
    outputs: [&[f32]; 2],
    stream: &Stream,
) -> Result<()> {
    for (layout, m, values) in [
        ("scalar", usize::try_from(sequence)?, outputs[0]),
        ("packed", usize::try_from(batch * sequence)?, outputs[1]),
    ] {
        // Explicit host hypothesis for the captured M3 Max shapes, not a runtime
        // selector.
        let mut partitions = if m == 8 {
            1
        } else {
            512 / (projection.output_features.div_ceil(32) * m.div_ceil(32))
        };
        partitions = partitions.max(1).min(projection.input_features / 32);
        while !projection.input_features.is_multiple_of(partitions * 32) {
            partitions -= 1;
        }
        let samples = reference::samples(projection, input, stream, partitions)?;
        let values = values
            .chunks_exact(projection.output_features)
            .flat_map(|row| {
                reference::columns(projection.output_features)
                    .into_iter()
                    .map(|column| f64::from(row[column]))
            })
            .collect::<Vec<_>>();
        writeln!(
            std::io::stderr().lock(),
            "mxfp4.split_control: {}",
            serde_json::json!({
                "label":label,"sequence":sequence,"layout":layout,"partitions":partitions,"samples":values.len(),
                "native_vs_bf16_partials":values.iter().zip(&samples.bf16_partials).filter(|(a,b)| a!=b).count(),
                "native_vs_fp32_partials":values.iter().zip(&samples.fp32_partials).filter(|(a,b)| a!=b).count(),
                "native_vs_bf16_tree":values.iter().zip(&samples.bf16_tree).filter(|(a,b)| a!=b).count(),
                "bf16_partials_vs_nearest":samples.bf16_partials.iter().zip(&samples.full).filter(|(a,b)| **a!=reference::nearest_bf16(**b)).count(),
                "fp32_partials_vs_nearest":samples.fp32_partials.iter().zip(&samples.full).filter(|(a,b)| **a!=reference::nearest_bf16(**b)).count()
            })
        )?;
    }
    Ok(())
}
