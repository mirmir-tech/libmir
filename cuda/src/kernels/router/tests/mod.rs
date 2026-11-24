use std::path::PathBuf;

use mircuda::{Compiler, Context, DeviceBuffer, DeviceElement, Driver, MemoryPool, Stream, bf16};
use models::semantic::{ActivationSpec, ExpertGroupLimit, RoutedExpertsSpec, RouterNormalization};

use super::{RouterUnitScoring, RouterUnitSpec, RouterUnitTopK, RouterUnitWeighting};
use crate::Result;

mod input_scale;
mod sparse_mixer;

const TOKENS: usize = 3;
const EXPERTS: usize = 64;
const TOP_K: usize = 6;

#[test]
fn renormalizes_the_selected_experts_and_scales_them() -> Result<()> {
    check(&routed(RouterNormalization::SoftmaxTopK, None, 2.5))
}

#[test]
fn weighs_by_the_probability_over_every_expert() -> Result<()> {
    check(&routed(RouterNormalization::SoftmaxProbabilityTopK, None, 1.0))
}

/// The bias moves the selection while the weights stay unbiased sigmoids.
#[test]
fn selects_by_biased_sigmoid_and_weighs_by_sigmoid() -> Result<()> {
    check(&routed(RouterNormalization::SigmoidBiasedTopK, None, 2.5))
}

#[test]
fn scales_probability_weights() -> Result<()> {
    check(&routed(RouterNormalization::SoftmaxProbabilityTopK, None, 16.0))
}

/// `group_limited_greedy`: the best expert ranks each group, the weights stay
/// probabilities over every expert.
#[test]
fn limits_probability_selection_to_the_best_groups() -> Result<()> {
    check(&routed(
        RouterNormalization::SoftmaxProbabilityTopK,
        Some(ExpertGroupLimit { groups: 8, selected: 3 }),
        1.0,
    ))
}

/// `noaux_tc`: the two best biased scores rank each group.
#[test]
fn limits_sigmoid_selection_to_the_best_groups() -> Result<()> {
    check(&routed(
        RouterNormalization::SigmoidBiasedTopK,
        Some(ExpertGroupLimit { groups: 8, selected: 4 }),
        2.5,
    ))
}

/// Llama 4: the top logits, each weighted by its own sigmoid.
#[test]
fn weighs_the_selected_logits_by_their_sigmoid() -> Result<()> {
    check(&routed(RouterNormalization::SigmoidInputScaledTopK, None, 1.0))
}

/// The routed experts of a router over this module's experts.
fn routed(
    router_normalization: RouterNormalization,
    group_limit: Option<ExpertGroupLimit>,
    output_scale: f64,
) -> RoutedExpertsSpec {
    RoutedExpertsSpec {
        expert_count: EXPERTS,
        top_k: TOP_K,
        intermediate_size: 1,
        activation: ActivationSpec::SwiGlu,
        router_normalization,
        group_limit,
        output_scale,
    }
}

/// The kernel's selection and weights against the `models` host routing.
fn check(experts: &RoutedExpertsSpec) -> Result<()> {
    let runtime = Runtime::new()?;
    let scoring = RouterUnitScoring::of(experts)?;
    let scores = (0..TOKENS * EXPERTS)
        .map(|index| {
            let value = u16::try_from((index * 37 + 11) % 101).unwrap_or(0);
            bf16::from_f32(f32::from(value) / 25.0 - 2.0)
        })
        .collect::<Vec<_>>();
    let spec = RouterUnitSpec {
        tokens: TOKENS,
        experts: EXPERTS,
        top_k: TOP_K,
    };
    // A bias of up to a whole sigmoid unit reorders the selection.
    let bias = (0..EXPERTS)
        .map(|expert| f32::from(u16::try_from((expert * 53 + 7) % 64).unwrap_or(0)) / 64.0)
        .collect::<Vec<_>>();
    let device_bias = runtime.copy(&bias)?;
    let sigmoid = scoring.weighting == RouterUnitWeighting::SigmoidBiased;
    let router = RouterUnitTopK::compile_scored(&runtime.compiler, spec, scoring)?;
    let device_scores = runtime.copy(&scores)?;
    let mut selected = runtime.pool.allocate::<u32>(&runtime.stream, TOKENS * TOP_K)?;
    let mut weights = runtime.pool.allocate::<bf16>(&runtime.stream, TOKENS * TOP_K)?;
    router.execute(
        &runtime.stream,
        &device_scores,
        sigmoid.then_some(&device_bias),
        &mut selected,
        &mut weights,
    )?;
    let (selected, weights) = (runtime.read(&selected)?, runtime.read(&weights)?);
    let mut limited = false;
    for token in 0..TOKENS {
        let logits = scores[token * EXPERTS..(token + 1) * EXPERTS]
            .iter()
            .map(|value| value.to_f32())
            .collect::<Vec<_>>();
        let route = experts.route(&logits, Some(&bias))?;
        let ungrouped = RoutedExpertsSpec { group_limit: None, ..experts.clone() };
        let indices = route.iter().map(|selected| selected.expert).collect::<Vec<_>>();
        let expected = route.iter().map(|selected| selected.weight);
        limited |= ungrouped.route(&logits, Some(&bias))? != route;
        let rows = token * TOP_K..(token + 1) * TOP_K;
        let actual = selected[rows.clone()].iter().map(|&index| index as usize).collect::<Vec<_>>();
        assert_eq!(actual, indices, "token {token}");
        for (weight, expected) in weights[rows].iter().zip(expected) {
            let error = (weight.to_f32() - expected).abs();
            assert!(
                error <= expected * 1e-2,
                "token {token}: {} against {expected}",
                weight.to_f32()
            );
        }
    }
    // Otherwise the groups would not be tested at all.
    assert_eq!(limited, experts.group_limit.is_some(), "groups change no selection");
    Ok(())
}

pub(super) struct Runtime {
    context: Context,
    pub(super) compiler: Compiler,
    pub(super) pool: MemoryPool,
    pub(super) stream: Stream,
}

impl Runtime {
    pub(super) fn new() -> Result<Self> {
        let driver = Driver::initialize()?;
        let device = driver.devices()?.into_iter().next().ok_or(mircuda::Error::InvalidLaunch)?;
        let context = driver.create_context(device)?;
        let stream = context.create_stream()?;
        let pool = context.default_memory_pool()?;
        let compiler = Compiler::with_include_paths(
            context.clone(),
            [PathBuf::from("/usr/local/cuda/include")],
        )?;
        Ok(Self { context, compiler, pool, stream })
    }

    pub(super) fn copy<T: DeviceElement + Copy>(&self, values: &[T]) -> Result<DeviceBuffer<T>> {
        let mut host = self.context.allocate_pinned::<T>(values.len())?;
        host.copy_from_slice(values)?;
        let mut device = self.pool.allocate::<T>(&self.stream, values.len())?;
        self.stream.copy_to_device(&mut host, &mut device)?;
        self.stream.synchronize()?;
        Ok(device)
    }

    pub(super) fn read<T: DeviceElement + Copy>(&self, device: &DeviceBuffer<T>) -> Result<Vec<T>> {
        let mut host = self.context.allocate_pinned::<T>(device.len())?;
        self.stream.copy_to_host(device, &mut host)?;
        self.stream.synchronize()?;
        Ok(host.to_vec()?)
    }
}
