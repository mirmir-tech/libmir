use mircuda::Stream;
use models::layout::PixelShuffleVisionConfig;

use super::{
    super::{
        super::CudaBackend,
        linear::{VisionLinear, required},
    },
    scratch::{PixelShuffleScratch, bf16, elements, single},
};
use crate::{
    CudaTensor, CudaTensorSet, Result,
    kernels::{
        PixelShuffleGeometry, PixelShuffleKernels, VisionElementwise, VisionElementwiseSpec,
    },
};

/// The encoder's optional final layer norm, the pixel shuffle and the
/// RMS-normalized squared-rectifier projector into the decoder's hidden size.
#[derive(Debug)]
pub(super) struct PixelShuffleProjector {
    final_norm: Option<(VisionElementwise, CudaTensor, CudaTensor)>,
    norm: CudaTensor,
    first: VisionLinear,
    output: VisionLinear,
    norm_eps: f32,
    hidden_size: usize,
}

impl PixelShuffleProjector {
    pub(super) fn new(
        backend: &CudaBackend,
        config: &PixelShuffleVisionConfig,
        tensors: &CudaTensorSet,
        geometry: PixelShuffleGeometry,
    ) -> Result<Self> {
        let soft = geometry.folded_tokens()?;
        let final_norm = if config.final_layer_norm {
            let spec = VisionElementwiseSpec {
                rows: geometry.tokens()?,
                columns: config.hidden_size,
                epsilon: single(config.layer_norm_eps)?,
            };
            Some((
                VisionElementwise::compile(&backend.inner.compiler, spec)?,
                required(tensors, "vision_projector.vision_final_layernorm.weight")?,
                required(tensors, "vision_projector.vision_final_layernorm.bias")?,
            ))
        } else {
            None
        };
        Ok(Self {
            final_norm,
            norm: required(tensors, "mlp1.0.weight")?,
            first: VisionLinear::new(
                backend,
                tensors,
                "mlp1.1",
                soft,
                geometry.folded_width()?,
                config.projector_hidden_size,
                false,
            )?,
            output: VisionLinear::new(
                backend,
                tensors,
                "mlp1.3",
                soft,
                config.projector_hidden_size,
                config.output_hidden_size,
                false,
            )?,
            norm_eps: single(config.projector_norm_eps)?,
            hidden_size: config.projector_hidden_size,
        })
    }

    pub(super) fn execute(
        &mut self,
        stream: &Stream,
        kernels: &PixelShuffleKernels,
        geometry: PixelShuffleGeometry,
        scratch: &mut PixelShuffleScratch,
    ) -> Result<()> {
        let encoded = match &self.final_norm {
            Some((norm, weight, bias)) => {
                norm.layer_norm(
                    stream,
                    &scratch.hidden_a,
                    bf16(weight)?,
                    bf16(bias)?,
                    &mut scratch.normalized,
                )?;
                &scratch.normalized
            },
            None => &scratch.hidden_a,
        };
        kernels.fold(stream, geometry, encoded, &mut scratch.folded)?;
        let soft = geometry.folded_tokens()?;
        kernels.rms_norm(
            stream,
            &scratch.folded,
            bf16(&self.norm)?,
            &mut scratch.folded_norm,
            soft,
            geometry.folded_width()?,
            self.norm_eps,
        )?;
        self.first.execute(&scratch.folded_norm, &mut scratch.projector_hidden)?;
        kernels.relu_square(
            stream,
            &scratch.projector_hidden,
            &mut scratch.projector_activation,
            elements(soft, self.hidden_size)?,
        )?;
        self.output.execute(&scratch.projector_activation, &mut scratch.output)
    }
}
