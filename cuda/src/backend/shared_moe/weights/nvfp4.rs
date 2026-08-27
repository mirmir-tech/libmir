use models::weights::{
    BlockActivationMode, BlockFormat, ExpertProjectionRole, RoutedExpertBindings, TensorBinding,
    TensorCatalog, TensorStorage,
};

use crate::{
    CudaBackend, Error, NvFp4ExpertBankConfig, NvFp4ExpertSource, NvFp4ScaleMode, Result,
    backend::block::experts::ExpertWeights,
};

pub(super) fn load(
    backend: &CudaBackend,
    catalog: &TensorCatalog,
    bindings: RoutedExpertBindings<'_>,
    experts: usize,
    hidden: usize,
    intermediate: usize,
) -> Result<ExpertWeights> {
    let activation_mode = mode(bindings)?;
    let gate = sources(catalog, bindings.individual(ExpertProjectionRole::Gate))?;
    let up = sources(catalog, bindings.individual(ExpertProjectionRole::Up))?;
    let down = sources(catalog, bindings.individual(ExpertProjectionRole::Down))?;
    let bank = |input, output, sources: &[NvFp4ExpertSource<'_>]| {
        backend.prepare_nvfp4_expert_bank(
            NvFp4ExpertBankConfig {
                experts,
                input_features: input,
                output_features: output,
            },
            sources,
        )
    };
    Ok(ExpertWeights::NvFp4 {
        gate: bank(hidden, intermediate, &gate)?,
        up: bank(hidden, intermediate, &up)?,
        down: bank(intermediate, hidden, &down)?,
        activation_mode,
    })
}

fn sources<'a>(
    catalog: &'a TensorCatalog,
    bindings: Vec<&TensorBinding>,
) -> Result<Vec<NvFp4ExpertSource<'a>>> {
    bindings.into_iter().map(|binding| source(catalog, binding)).collect()
}

fn source<'a>(
    catalog: &'a TensorCatalog,
    binding: &TensorBinding,
) -> Result<NvFp4ExpertSource<'a>> {
    let TensorStorage::BlockQuantized {
        format,
        scales,
        global_scale: Some(global_scale),
        input_scale: Some(input_scale),
        ..
    } = &binding.storage
    else {
        return Err(Error::UnsupportedDecoderLayer(format!(
            "CUDA individual expert is not complete NVFP4: {}",
            binding.source
        )));
    };
    if format.format != BlockFormat::NvFp4 {
        return Err(Error::UnsupportedDecoderLayer(format!(
            "CUDA individual expert is not NVFP4: {}",
            binding.source
        )));
    }
    let get = |name: &str| catalog.get(name).ok_or_else(|| Error::MissingTensor(name.into()));
    Ok(NvFp4ExpertSource {
        weight: get(&binding.source)?,
        weight_scale: get(scales)?,
        weight_scale_2: get(global_scale)?,
        input_scale: get(input_scale)?,
        scale_mode: NvFp4ScaleMode::from_names(global_scale, input_scale)?,
    })
}

fn mode(bindings: RoutedExpertBindings<'_>) -> Result<BlockActivationMode> {
    let mut mode = None;
    let bindings =
        [ExpertProjectionRole::Gate, ExpertProjectionRole::Up, ExpertProjectionRole::Down]
            .into_iter()
            .flat_map(|projection| bindings.individual(projection));
    for binding in bindings {
        let TensorStorage::BlockQuantized { format, .. } = binding.storage else {
            return Err(Error::UnsupportedDecoderLayer(
                "CUDA individual experts mix non-block storage".into(),
            ));
        };
        if format.format != BlockFormat::NvFp4
            || mode
                .replace(format.activation_mode)
                .is_some_and(|value| value != format.activation_mode)
        {
            return Err(Error::UnsupportedDecoderLayer(
                "CUDA individual experts mix NVFP4 activation contracts".into(),
            ));
        }
    }
    mode.ok_or_else(|| Error::UnsupportedDecoderLayer("CUDA has no individual experts".into()))
}
