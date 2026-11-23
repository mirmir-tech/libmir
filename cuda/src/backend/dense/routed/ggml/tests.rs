//! Per-expert output scales of the routed GGML banks against banks whose
//! blocks carry the same factors (powers of two keep both exact).

use models::gguf::GgmlType;

use super::{GgmlExpertGeometry, GgmlRoutedExperts};
use crate::{
    CudaBackend, CudaConfig, GatedActivation, GgmlCheckpointWeight, Result,
    backend::{
        dense::{RoutedExpertsConfig, routed::weights::GgmlExpertWeights},
        linear::ggml_tests::{assert_close, copy, input, q8_0, read},
    },
    kernels::GgmlKernelType,
};

const EXPERTS: usize = 3;
const HIDDEN: usize = 256;
const INTERMEDIATE: usize = 64;
const SCALES: [[f32; EXPERTS]; 3] = [[2.0, 0.5, -1.0], [0.5, -2.0, 1.0], [-0.5, 4.0, 0.25]];

#[test]
fn output_scales_multiply_each_experts_products() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let shapes = [(INTERMEDIATE, HIDDEN), (INTERMEDIATE, HIDDEN), (HIDDEN, INTERMEDIATE)];
    let banks = |factors: &[[f32; EXPERTS]; 3], scaled: bool| -> Result<GgmlExpertWeights> {
        let mut weights = Vec::with_capacity(3);
        for ((rows, columns), factors) in shapes.into_iter().zip(factors) {
            let folded = if scaled {
                &[][..]
            } else {
                &factors[..]
            };
            let blocks = copy(&backend, &q8_0(EXPERTS, rows, columns, folded))?;
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
        Ok(GgmlExpertWeights::separate([gate, up, down], (HIDDEN, INTERMEDIATE)))
    };
    let (folded, scaled) = (banks(&SCALES, false)?, banks(&SCALES, true)?);
    let geometry = GgmlExpertGeometry {
        hidden: HIDDEN,
        intermediate: INTERMEDIATE,
        experts: RoutedExpertsConfig { count: EXPERTS, top_k: 2 },
        activation: GatedActivation::Silu,
    };
    // Two rows take the gathered GEMVs, forty the grouped int8 GEMMs.
    for rows in [2, 40] {
        let selected = (0..rows)
            .flat_map(|row| [row % EXPERTS, (row + 1) % EXPERTS])
            .map(u32::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let weights = (0..rows)
            .flat_map(|_| [mircuda::bf16::from_f32(0.75), mircuda::bf16::from_f32(0.25)])
            .collect::<Vec<_>>();
        let (selected, weights) = (copy(&backend, &selected)?, copy(&backend, &weights)?);
        let x = copy(&backend, &input(rows * HIDDEN))?;
        let mut experts = GgmlRoutedExperts::new(&backend, geometry, rows)?;
        let allocate = || backend.inner.pool.allocate(&backend.inner.stream, rows * HIDDEN);
        let (mut expected, mut actual) = (allocate()?, allocate()?);
        experts.execute(&x, &selected, &weights, &folded, &mut expected)?;
        experts.execute(&x, &selected, &weights, &scaled, &mut actual)?;
        let label = format!("{rows} rows");
        assert_close(&read(&backend, &expected)?, &read(&backend, &actual)?, &label);
    }
    Ok(())
}
