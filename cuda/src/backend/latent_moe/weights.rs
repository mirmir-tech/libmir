use models::weights::{ExpertProjectionRole, LatentRoutedBindings, TensorCatalog};

use super::LatentMoeConfig;
use crate::{
    CudaBackend, CudaTensor, CudaTensorSet, Error, NvFp4ExpertBankConfig, NvFp4ExpertSource,
    Result,
    backend::linear::{CheckpointProjectionWeight, MarlinNvFp4Bank},
};

/// Checkpoint weights of one latent routed-expert block.
#[derive(Clone, Debug)]
pub struct LatentMoeWeights {
    pub(super) router: CudaTensor,
    pub(super) correction_bias: CudaTensor,
    pub(super) latent_input: CheckpointProjectionWeight,
    pub(super) latent_output: CheckpointProjectionWeight,
    /// Experts stay resident only in the Marlin layout; the checkpoint-layout
    /// banks are released as soon as they are repacked.
    pub(super) up: MarlinNvFp4Bank,
    pub(super) down: MarlinNvFp4Bank,
    pub(super) shared: Option<SharedExpertWeights>,
}

#[derive(Clone, Debug)]
pub(super) struct SharedExpertWeights {
    pub(super) up: CheckpointProjectionWeight,
    pub(super) down: CheckpointProjectionWeight,
}

impl LatentMoeWeights {
    /// Loads the block from typed bindings. Dense tensors come from
    /// `tensors`; individual NVFP4 experts are banked from `catalog`.
    pub fn load_bindings(
        backend: &CudaBackend,
        tensors: &CudaTensorSet,
        catalog: &TensorCatalog,
        bindings: LatentRoutedBindings<'_>,
        config: LatentMoeConfig,
    ) -> Result<Self> {
        let tensor = |name: &str| {
            tensors.get(name).cloned().ok_or_else(|| Error::MissingTensor(name.into()))
        };
        let bank = |projection, input, output| {
            let sources = bindings
                .routed
                .individual(projection)
                .into_iter()
                .map(|binding| NvFp4ExpertSource::from_binding(catalog, binding))
                .collect::<Result<Vec<_>>>()?;
            backend
                .prepare_nvfp4_expert_bank(
                    NvFp4ExpertBankConfig {
                        experts: config.experts,
                        input_features: input,
                        output_features: output,
                    },
                    &sources,
                )?
                .marlin(backend)
        };
        let projection =
            |binding| CheckpointProjectionWeight::load_binding_prepared(backend, tensors, binding);
        Ok(Self {
            router: tensor(&bindings.router.source)?,
            correction_bias: tensor(&bindings.correction_bias.source)?,
            latent_input: projection(bindings.latent_input)?,
            latent_output: projection(bindings.latent_output)?,
            up: bank(ExpertProjectionRole::Up, config.latent, config.expert_intermediate)?,
            down: bank(ExpertProjectionRole::Down, config.expert_intermediate, config.latent)?,
            shared: bindings
                .shared
                .map(|shared| {
                    Ok::<_, Error>(SharedExpertWeights {
                        up: projection(shared.up)?,
                        down: projection(shared.down)?,
                    })
                })
                .transpose()?,
        })
    }

    pub(super) fn validate(&self, config: LatentMoeConfig) -> Result<()> {
        let router = self
            .router
            .as_bf16()
            .is_some_and(|router| router.len() == config.experts * config.hidden);
        let bias = self.correction_bias.as_f32().is_some_and(|bias| bias.len() == config.experts);
        let banks = self.up.config()
            == NvFp4ExpertBankConfig {
                experts: config.experts,
                input_features: config.latent,
                output_features: config.expert_intermediate,
            }
            && self.down.config()
                == NvFp4ExpertBankConfig {
                    experts: config.experts,
                    input_features: config.expert_intermediate,
                    output_features: config.latent,
                };
        if router && bias && banks && self.shared.is_some() == config.shared_intermediate.is_some()
        {
            Ok(())
        } else {
            Err(Error::InvalidDecoderKernel(
                "latent routed-expert weights differ from configuration",
            ))
        }
    }
}
