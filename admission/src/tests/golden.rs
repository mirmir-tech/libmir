//! Golden admission over every fixture: a change in selection shows up as a
//! diff here.

use models::{
    execution::TaskExecutionPlan,
    semantic::{ActivationSpec, FeedForwardSpec, SemanticModelSpec},
};

use super::{
    fixtures::{
        AnyResult, clamped_routed, dense_and_routed, dense_contract, mixed_unsupported,
        shared_routed, unshared_routed,
    },
    latent::latent_contract,
    state_space::hybrid_contract,
};
use crate::{
    ArchitectureError,
    cuda::{self, CudaDecoderRuntime as Cuda},
    metal::{self, MetalDecoderRuntime as Metal},
};

type Fixture = (&'static str, TaskExecutionPlan, SemanticModelSpec);

pub(super) fn fixtures() -> AnyResult<Vec<Fixture>> {
    let (task, dense) = dense_contract()?;
    // Linear attention alone, which Metal runs and CUDA pairs with softmax.
    let mut linear = shared_routed(dense.clone());
    linear.decoder.layers.truncate(1);
    let mut mixed = shared_routed(dense.clone());
    for (layer, original) in mixed.decoder.layers.iter_mut().zip(&dense.decoder.layers) {
        layer.feed_forward = original.feed_forward.clone();
    }
    let (latent_task, latent) = latent_contract("softmax")?;
    let (sigmoid_task, sigmoid) = latent_contract("sigmoid")?;
    let (hybrid_task, hybrid) = hybrid_contract()?;
    // Gemma 4's dense layers: gated GELU behind value-normalized attention.
    let mut gelu = dense_and_routed(dense.clone());
    for layer in &mut gelu.decoder.layers {
        layer.feed_forward = FeedForwardSpec::Dense {
            intermediate_size: 16,
            activation: ActivationSpec::GeluTanh,
        };
    }
    // Gemma 4's E models: the dense GELU layers with per-layer inputs.
    let mut per_layer = gelu.clone();
    per_layer.decoder.per_layer_input =
        Some(models::semantic::PerLayerInputSpec { width: 4, vocab_size: 32 });
    // ... whose later layers attend over an earlier layer's keys and values.
    let mut sharing = per_layer.clone();
    if let Some(models::semantic::MixerSpec::SoftmaxAttention(attention)) =
        sharing.decoder.layers.get_mut(1).map(|layer| &mut layer.mixer)
    {
        attention.key_value_source = Some(0);
    }
    Ok(vec![
        ("dense", task.clone(), dense.clone()),
        ("dense and routed", task.clone(), dense_and_routed(dense.clone())),
        ("dense gelu", task.clone(), gelu),
        ("per-layer inputs", task.clone(), per_layer),
        ("shared key values", task.clone(), sharing),
        ("shared routed", task.clone(), shared_routed(dense.clone())),
        ("dense mixed", task.clone(), mixed),
        ("linear shared routed", task.clone(), linear),
        ("clamped routed", task.clone(), clamped_routed(dense.clone())),
        ("unshared routed", task.clone(), unshared_routed(dense.clone())),
        ("unsupported mix", task, mixed_unsupported(dense)),
        ("latent softmax", latent_task, latent),
        ("latent sigmoid", sigmoid_task, sigmoid),
        ("state-space hybrid", hybrid_task, hybrid),
    ])
}

/// The admitted runtime, or `None` when no runtime executes the plan.
fn cuda_runtime(task: &TaskExecutionPlan, spec: &SemanticModelSpec) -> AnyResult<Option<Cuda>> {
    match cuda::admit(task, Some(spec)) {
        Ok(architecture) => Ok(architecture.runtime()),
        Err(ArchitectureError::NoRuntime { .. }) => Ok(None),
        Err(error) => Err(error.into()),
    }
}

fn metal_runtime(task: &TaskExecutionPlan, spec: &SemanticModelSpec) -> AnyResult<Option<Metal>> {
    match metal::admit(task, Some(spec)) {
        Ok(architecture) => Ok(architecture.runtime()),
        Err(ArchitectureError::NoRuntime { .. }) => Ok(None),
        Err(error) => Err(error.into()),
    }
}

#[test]
fn admits_each_fixture_to_its_runtime() -> AnyResult<()> {
    let expected = [
        ("dense", Some(Cuda::Dense), Some(Metal::Dense)),
        ("dense and routed", Some(Cuda::DenseAndRouted), Some(Metal::DenseAndRouted)),
        ("dense gelu", Some(Cuda::Dense), Some(Metal::DenseGelu)),
        ("per-layer inputs", Some(Cuda::Dense), Some(Metal::DenseGelu)),
        ("shared key values", Some(Cuda::Dense), Some(Metal::DenseGelu)),
        ("shared routed", Some(Cuda::SharedRouted), Some(Metal::SharedRouted)),
        ("dense mixed", Some(Cuda::DenseMixed), Some(Metal::DenseMixed)),
        ("linear shared routed", None, Some(Metal::SharedRouted)),
        ("clamped routed", Some(Cuda::ClampedRouted), Some(Metal::ClampedRouted)),
        // Q/K-normalized routed experts run in the graph decoder on CUDA;
        // Metal's dense runtime runs them too.
        ("unshared routed", Some(Cuda::Dense), Some(Metal::Dense)),
        ("unsupported mix", None, None),
        ("latent softmax", Some(Cuda::LatentAttention), Some(Metal::LatentAttention)),
        ("latent sigmoid", Some(Cuda::LatentAttention), Some(Metal::LatentAttention)),
        (
            "state-space hybrid",
            Some(Cuda::StateSpaceHybrid),
            Some(Metal::StateSpaceHybrid),
        ),
    ];
    let fixtures = fixtures()?;
    assert_eq!(fixtures.len(), expected.len());
    for ((name, task, spec), (expected_name, cuda, metal)) in fixtures.iter().zip(expected) {
        assert_eq!(*name, expected_name);
        assert_eq!(cuda_runtime(task, spec)?, cuda, "CUDA {name}");
        assert_eq!(metal_runtime(task, spec)?, metal, "Metal {name}");
    }
    Ok(())
}

/// Several runtimes may execute one plan; distinct priorities keep the
/// choice from depending on declaration order.
#[test]
fn no_two_runtimes_share_a_priority() {
    for (index, runtime) in Cuda::ALL.iter().enumerate() {
        assert!(
            Cuda::ALL[index + 1..]
                .iter()
                .all(|other| other.priority() != runtime.priority())
        );
    }
    for (index, runtime) in Metal::ALL.iter().enumerate() {
        assert!(
            Metal::ALL[index + 1..]
                .iter()
                .all(|other| other.priority() != runtime.priority())
        );
    }
}

/// Each runtime's table covers the fixture it exists for.
#[test]
fn each_runtime_covers_its_fixture() -> AnyResult<()> {
    let fixtures = fixtures()?;
    let find = |name: &str| fixtures.iter().find(|fixture| fixture.0 == name);
    let cuda_cases = [
        (Cuda::Dense, "dense"),
        (Cuda::DenseAndRouted, "dense and routed"),
        (Cuda::SharedRouted, "shared routed"),
        (Cuda::DenseMixed, "dense mixed"),
        (Cuda::ClampedRouted, "clamped routed"),
        (Cuda::StateSpaceHybrid, "state-space hybrid"),
        (Cuda::LatentAttention, "latent sigmoid"),
    ];
    for (runtime, name) in cuda_cases {
        let (_, task, spec) = find(name).ok_or(name)?;
        let TaskExecutionPlan::Generation { decoder } = task else {
            continue;
        };
        let plan = cuda::CudaDecoderPlan::lower(spec, decoder.attention_scale)?;
        assert_eq!(runtime.capabilities().missing(&plan.requirements()), [], "{runtime:?}");
    }
    let metal_cases = [
        (Metal::Dense, "dense"),
        (Metal::DenseAndRouted, "dense and routed"),
        (Metal::DenseGelu, "dense gelu"),
        (Metal::SharedRouted, "shared routed"),
        (Metal::DenseMixed, "dense mixed"),
        (Metal::ClampedRouted, "clamped routed"),
        (Metal::StateSpaceHybrid, "state-space hybrid"),
        (Metal::LatentAttention, "latent softmax"),
    ];
    for (runtime, name) in metal_cases {
        let (_, task, spec) = find(name).ok_or(name)?;
        let TaskExecutionPlan::Generation { decoder } = task else {
            continue;
        };
        let plan = metal::plan(decoder, spec)?;
        assert_eq!(runtime.capabilities().missing(&plan.requirements()), [], "{runtime:?}");
    }
    Ok(())
}
