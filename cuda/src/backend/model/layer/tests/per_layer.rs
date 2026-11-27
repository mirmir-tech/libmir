//! Gemma 4 layers whose per-layer input gates store their projections
//! differently.

use models::gguf::GgmlType;

use super::{HIDDEN, TOKENS, config, session, shared_prefill, tensor};
use crate::{
    CudaBackend, CudaConfig, DenseDownSource, DenseFeedForwardSource, DenseGateUpSource,
    DenseOutputSource, DenseQkvSource, DenseWeightSource, Error, GgmlCheckpointWeight,
    PerLayerInputWeights, ProjectionFormat, Result,
    backend::{
        linear::{
            CheckpointProjectionWeight,
            ggml_tests::{copy_blocks, q8_0},
        },
        model::layer::SessionLayer,
    },
    kernels::GgmlKernelType,
};

fn projection(
    backend: &CudaBackend,
    ggml: bool,
    seed: usize,
) -> Result<CheckpointProjectionWeight> {
    Ok(if ggml {
        CheckpointProjectionWeight::Ggml(GgmlCheckpointWeight::from_blocks(
            copy_blocks(backend, &q8_0(1, HIDDEN, HIDDEN, &[]))?,
            GgmlKernelType::new(GgmlType::Q8_0)?,
            (1, HIDDEN, HIDDEN),
            None,
        ))
    } else {
        CheckpointProjectionWeight::Dense(tensor(backend, "per_layer", &[HIDDEN, HIDDEN], seed)?)
    })
}

/// A BF16 Gemma-style layer `index` whose per-layer gate and projection are
/// GGML blocks or BF16.
fn layer(backend: &CudaBackend, index: usize, ggml: bool) -> Result<SessionLayer> {
    let square = |seed| tensor(backend, "weight", &[HIDDEN, HIDDEN], seed);
    let norm = tensor(backend, "norm", &[HIDDEN], 3)?;
    let qkv = backend.pack_bf16_linears([&square(1)?, &square(2)?, &square(5)?])?;
    let gate_up = backend.pack_bf16_linear_pair(&square(7)?, &square(11)?)?;
    let (output, down) = (square(13)?, square(17)?);
    let template = backend.prepare_dense_swiglu_layer_template(
        config(index, ProjectionFormat::Bf16, true),
        DenseWeightSource {
            input_norm: Some(&norm),
            qkv: DenseQkvSource::Bf16(&qkv),
            query_norm: None,
            key_norm: None,
            output: DenseOutputSource::Bf16(&output),
            post_attention_norm: Some(&norm),
            sandwich: None,
            feed_forward: DenseFeedForwardSource::Gated {
                gate_up: DenseGateUpSource::Bf16(&gate_up),
                down: DenseDownSource::Bf16(&down),
            },
            per_layer_input: Some(PerLayerInputWeights {
                gate: projection(backend, ggml, 19)?,
                projection: projection(backend, ggml, 23)?,
                output_norm: norm.clone(),
            }),
        },
    )?;
    session(backend, template)
}

/// Layers whose per-layer projections share a layout share a prefill plan;
/// a GGML layer gets its own, and a plan refuses the other layout's weights
/// rather than reading them as its own.
#[test]
fn per_layer_layouts_split_shared_prefill_plans() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let dense = layer(&backend, 0, false)?;
    let mut other_dense = layer(&backend, 1, false)?;
    let mut ggml = layer(&backend, 2, true)?;
    assert_eq!(dense.prefill_signature(), other_dense.prefill_signature());
    assert_ne!(dense.prefill_signature(), ggml.prefill_signature());
    let mut dense_plan = dense.instantiate_shared_prefill(TOKENS)?;
    let mut ggml_plan = ggml.instantiate_shared_prefill(TOKENS)?;
    shared_prefill(&backend, &mut dense_plan, &mut other_dense)?;
    shared_prefill(&backend, &mut ggml_plan, &mut ggml)?;
    let refused = shared_prefill(&backend, &mut dense_plan, &mut ggml);
    assert!(matches!(refused, Err(Error::InvalidExecutionPlan(_))), "{refused:?}");
    Ok(())
}
