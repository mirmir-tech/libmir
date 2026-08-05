use mircuda::bf16;
use models::weights::{TensorCatalog, TensorInfo};

use super::support::{catalog, copy, read, values};
use crate::{
    CudaBackend, CudaConfig, Error, GatedActivation, NvFp4ExpertBank, NvFp4ExpertBankConfig,
    NvFp4ExpertSource, Result, kernels::SelectedNvFp4TiledRows,
};

const SELECTED: usize = 8;
const BASES: [&str; 2] =
    ["model.language_model.layers.0.mlp.experts", "model.language_model.layers.0.experts"];

#[test]
fn checkpoint_tiled_w4a16_matches_reference() -> Result<()> {
    let Some(catalog) = catalog()? else {
        return Ok(());
    };
    let backend = CudaBackend::new(CudaConfig::default())?;
    let (hidden, intermediate) = geometry(&catalog)?;
    let mut reference =
        prepare(&backend, &catalog, hidden, intermediate, CandidateKind::Reference)?;
    let mut candidates =
        [SelectedNvFp4TiledRows::Two, SelectedNvFp4TiledRows::Four, SelectedNvFp4TiledRows::Eight]
            .into_iter()
            .map(|rows| {
                prepare(&backend, &catalog, hidden, intermediate, CandidateKind::Tiled(rows))
            })
            .collect::<Result<Vec<_>>>()?;
    let input = copy(&backend, &values(hidden)?)?;
    let selected = copy(&backend, &(0..u32::try_from(SELECTED)?).collect::<Vec<_>>())?;
    let routing = copy(&backend, &[bf16::from_f32(0.125); SELECTED])?;
    let mut expected_output = backend.inner.pool.allocate(&backend.inner.stream, hidden)?;
    reference.execute(&input, &selected, &routing, &mut expected_output)?;
    let expected = read(&backend, &expected_output)?;

    for candidate in &mut candidates {
        let mut output = backend.inner.pool.allocate(&backend.inner.stream, hidden)?;
        candidate.execute(&input, &selected, &routing, &mut output)?;
        assert_eq!(read(&backend, &output)?, expected);
    }
    Ok(())
}

#[test]
fn checkpoint_tensor_core_w4a16_matches_reference() -> Result<()> {
    let Some(catalog) = catalog()? else {
        return Ok(());
    };
    let backend = CudaBackend::new(CudaConfig::default())?;
    let (hidden, intermediate) = geometry(&catalog)?;
    let mut reference =
        prepare(&backend, &catalog, hidden, intermediate, CandidateKind::Reference)?;
    let mut candidate =
        prepare(&backend, &catalog, hidden, intermediate, CandidateKind::TensorCore)?;
    let input = copy(&backend, &values(hidden)?)?;
    let selected = copy(&backend, &(0..u32::try_from(SELECTED)?).collect::<Vec<_>>())?;
    let routing = copy(&backend, &[bf16::from_f32(0.125); SELECTED])?;
    let mut expected_output = backend.inner.pool.allocate(&backend.inner.stream, hidden)?;
    let mut actual_output = backend.inner.pool.allocate(&backend.inner.stream, hidden)?;
    reference.execute(&input, &selected, &routing, &mut expected_output)?;
    candidate.execute(&input, &selected, &routing, &mut actual_output)?;
    let expected = read(&backend, &expected_output)?;
    let actual = read(&backend, &actual_output)?;
    for (index, (expected, actual)) in expected.iter().zip(actual).enumerate() {
        let expected = expected.to_f32();
        let tolerance = (expected.abs() * 0.12).max(2.0);
        assert!((actual.to_f32() - expected).abs() <= tolerance, "output {index}");
    }
    Ok(())
}

#[test]
fn checkpoint_marlin_w4a16_matches_reference() -> Result<()> {
    let Some(catalog) = catalog()? else {
        return Ok(());
    };
    let backend = CudaBackend::new(CudaConfig::default())?;
    let (hidden, intermediate) = geometry(&catalog)?;
    let mut reference =
        prepare(&backend, &catalog, hidden, intermediate, CandidateKind::Reference)?;
    let mut candidates = [
        mircuda::MarlinNvFp4ThreadConfig::N128K128,
        mircuda::MarlinNvFp4ThreadConfig::N128K64,
        mircuda::MarlinNvFp4ThreadConfig::N64K128,
    ]
    .into_iter()
    .map(|config| prepare(&backend, &catalog, hidden, intermediate, CandidateKind::Marlin(config)))
    .collect::<Result<Vec<_>>>()?;
    let input = copy(&backend, &values(hidden)?)?;
    let selected = copy(&backend, &(0..u32::try_from(SELECTED)?).collect::<Vec<_>>())?;
    let routing = copy(&backend, &[bf16::from_f32(0.125); SELECTED])?;
    let mut expected_output = backend.inner.pool.allocate(&backend.inner.stream, hidden)?;
    reference.execute(&input, &selected, &routing, &mut expected_output)?;
    let expected = read(&backend, &expected_output)?;
    for candidate in &mut candidates {
        let mut output = backend.inner.pool.allocate(&backend.inner.stream, hidden)?;
        candidate.execute(&input, &selected, &routing, &mut output)?;
        for (index, (expected, actual)) in expected.iter().zip(read(&backend, &output)?).enumerate()
        {
            let expected = expected.to_f32();
            let tolerance = (expected.abs() * 0.03).max(0.5);
            assert!((actual.to_f32() - expected).abs() <= tolerance, "output {index}");
        }
    }
    Ok(())
}

fn prepare(
    backend: &CudaBackend,
    catalog: &TensorCatalog,
    hidden: usize,
    intermediate: usize,
    kind: CandidateKind,
) -> Result<Plan> {
    let gate = bank(backend, catalog, "gate_proj", hidden, intermediate)?;
    let up = bank(backend, catalog, "up_proj", hidden, intermediate)?;
    let down = bank(backend, catalog, "down_proj", intermediate, hidden)?;
    match kind {
        CandidateKind::Tiled(rows) => backend
            .prepare_tiled_selected_nvfp4_moe_bf16(
                1,
                SELECTED,
                GatedActivation::GeluTanh,
                rows,
                [gate, up, down],
            )
            .map(Plan::Tiled),
        CandidateKind::TensorCore => backend
            .prepare_selected_nvfp4_weight_only_tensor_core_moe_bf16(
                1,
                SELECTED,
                GatedActivation::GeluTanh,
                [gate, up, down],
            )
            .map(Plan::TensorCore),
        CandidateKind::Marlin(config) => backend
            .prepare_marlin_nvfp4_moe_bf16(
                1,
                SELECTED,
                GatedActivation::GeluTanh,
                config,
                [gate, up, down],
            )
            .map(Plan::Marlin),
        CandidateKind::Reference => backend
            .prepare_selected_nvfp4_moe_bf16(SELECTED, GatedActivation::GeluTanh, gate, up, down)
            .map(Plan::Reference),
    }
}

#[derive(Clone, Copy)]
enum CandidateKind {
    Reference,
    TensorCore,
    Marlin(mircuda::MarlinNvFp4ThreadConfig),
    Tiled(SelectedNvFp4TiledRows),
}

enum Plan {
    Reference(super::SelectedNvFp4MoeBf16),
    TensorCore(super::SelectedNvFp4WeightOnlyTensorCoreMoeBf16),
    Marlin(super::MarlinNvFp4MoeBf16),
    Tiled(super::TiledSelectedNvFp4MoeBf16),
}

impl Plan {
    fn execute(
        &mut self,
        input: &mircuda::DeviceBuffer<bf16>,
        selected: &mircuda::DeviceBuffer<u32>,
        routing: &mircuda::DeviceBuffer<bf16>,
        output: &mut mircuda::DeviceBuffer<bf16>,
    ) -> Result<()> {
        match self {
            Self::Reference(plan) => plan.execute(input, selected, routing, output),
            Self::TensorCore(plan) => plan.execute(input, selected, routing, output),
            Self::Marlin(plan) => plan.execute(input, selected, routing, output),
            Self::Tiled(plan) => plan.execute(input, selected, routing, output),
        }
    }
}

fn geometry(catalog: &TensorCatalog) -> Result<(usize, usize)> {
    let gate = required(catalog, &format!("{}.0.gate_proj.weight", base(catalog)?))?;
    Ok((gate.shape[1] * 2, gate.shape[0]))
}

fn bank(
    backend: &CudaBackend,
    catalog: &TensorCatalog,
    projection: &str,
    input: usize,
    output: usize,
) -> Result<NvFp4ExpertBank> {
    let base = base(catalog)?;
    let sources = (0..SELECTED)
        .map(|expert| source(catalog, &format!("{base}.{expert}.{projection}")))
        .collect::<Result<Vec<_>>>()?;
    backend.prepare_nvfp4_expert_bank(
        NvFp4ExpertBankConfig {
            experts: SELECTED,
            input_features: input,
            output_features: output,
        },
        &sources,
    )
}

fn source<'a>(catalog: &'a TensorCatalog, prefix: &str) -> Result<NvFp4ExpertSource<'a>> {
    Ok(NvFp4ExpertSource {
        weight: required(catalog, &format!("{prefix}.weight"))?,
        weight_scale: required(catalog, &format!("{prefix}.weight_scale"))?,
        weight_scale_2: required(catalog, &format!("{prefix}.weight_scale_2"))?,
        input_scale: required(catalog, &format!("{prefix}.input_scale"))?,
    })
}

fn base(catalog: &TensorCatalog) -> Result<&'static str> {
    BASES
        .into_iter()
        .find(|base| catalog.tensors.iter().any(|tensor| tensor.name.starts_with(base)))
        .ok_or_else(|| Error::MissingTensor("layer 0 routed experts".into()))
}

fn required<'a>(catalog: &'a TensorCatalog, name: &str) -> Result<&'a TensorInfo> {
    catalog
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .ok_or_else(|| Error::MissingTensor(name.into()))
}
