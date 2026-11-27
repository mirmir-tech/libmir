//! Per-expert output scales of the routed GGML banks against banks whose
//! blocks carry the same factors (powers of two keep both exact).

use models::gguf::GgmlType;

use super::{GgmlExpertGeometry, GgmlRoutedExperts};
use crate::{
    CudaBackend, CudaConfig, GatedActivation, GgmlCheckpointWeight, Result,
    backend::{
        dense::{RoutedExpertsConfig, routed::weights::GgmlExpertWeights},
        linear::ggml_tests::{assert_close, copy, copy_blocks, input, q8_0, read},
    },
    kernels::GgmlKernelType,
};

const EXPERTS: usize = 3;
const SCALES: [[f32; EXPERTS]; 3] = [[2.0, 0.5, -1.0], [0.5, -2.0, 1.0], [-0.5, 4.0, 0.25]];

#[test]
fn output_scales_multiply_each_experts_products() -> Result<()> {
    check((256, 64), &[2, 40])
}

/// The backend interns quantized activations by shape, so equal hidden and
/// expert widths (gpt-oss: 2880 and 2880) hand the grouped prefill path one
/// activation buffer for both projections.
#[test]
fn grouped_prefill_takes_equal_hidden_and_expert_widths() -> Result<()> {
    check((256, 256), &[40])
}

/// Banks with folded factors against the same banks with output scales, for
/// each row count: two rows take the gathered GEMVs, forty the grouped int8
/// GEMMs.
fn check((hidden, intermediate): (usize, usize), row_counts: &[usize]) -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let shapes = [(intermediate, hidden), (intermediate, hidden), (hidden, intermediate)];
    let banks = |factors: &[[f32; EXPERTS]; 3], scaled: bool| -> Result<GgmlExpertWeights> {
        let mut weights = Vec::with_capacity(3);
        for ((rows, columns), factors) in shapes.into_iter().zip(factors) {
            let folded = if scaled {
                &[][..]
            } else {
                &factors[..]
            };
            let blocks = copy_blocks(&backend, &q8_0(EXPERTS, rows, columns, folded))?;
            let scale = scaled.then(|| copy(&backend, factors)).transpose()?;
            let geometry = (EXPERTS, rows, columns);
            weights.push(GgmlCheckpointWeight::from_blocks(
                blocks,
                GgmlKernelType::new(GgmlType::Q8_0)?,
                geometry,
                scale,
            ));
        }
        let [gate, up, down]: [GgmlCheckpointWeight; 3] = weights
            .try_into()
            .map_err(|_| crate::Error::InvalidDecoderKernel("three banks"))?;
        Ok(GgmlExpertWeights::separate([gate, up, down], (hidden, intermediate)))
    };
    let (folded, scaled) = (banks(&SCALES, false)?, banks(&SCALES, true)?);
    let geometry = GgmlExpertGeometry {
        hidden,
        intermediate,
        experts: RoutedExpertsConfig { count: EXPERTS, top_k: 2 },
        activation: GatedActivation::Silu,
    };
    for &rows in row_counts {
        let selected = (0..rows)
            .flat_map(|row| [row % EXPERTS, (row + 1) % EXPERTS])
            .map(u32::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let weights = (0..rows)
            .flat_map(|_| [mircuda::bf16::from_f32(0.75), mircuda::bf16::from_f32(0.25)])
            .collect::<Vec<_>>();
        let (selected, weights) = (copy(&backend, &selected)?, copy(&backend, &weights)?);
        let x = copy(&backend, &input(rows * hidden))?;
        let mut experts = GgmlRoutedExperts::new(&backend, geometry, rows)?;
        let allocate = || backend.inner.pool.allocate(&backend.inner.stream, rows * hidden);
        let (mut expected, mut actual) = (allocate()?, allocate()?);
        experts.execute(&x, &selected, &weights, &folded, &mut expected)?;
        experts.execute(&x, &selected, &weights, &scaled, &mut actual)?;
        let label = format!("{rows} rows of {hidden} by {intermediate}");
        assert_close(&read(&backend, &expected)?, &read(&backend, &actual)?, &label);
    }
    Ok(())
}
