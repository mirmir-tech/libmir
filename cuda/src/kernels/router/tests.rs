use std::path::PathBuf;

use mircuda::{Compiler, Context, DeviceBuffer, DeviceElement, Driver, MemoryPool, Stream, bf16};

use super::{RouterUnitScoring, RouterUnitSpec, RouterUnitTopK, RouterUnitWeighting};
use crate::Result;

const TOKENS: usize = 3;
const EXPERTS: usize = 64;
const TOP_K: usize = 6;

#[test]
fn renormalizes_the_selected_experts_and_scales_them() -> Result<()> {
    check(RouterUnitScoring {
        weighting: RouterUnitWeighting::Renormalized,
        scale: 2.5,
    })
}

#[test]
fn weighs_by_the_probability_over_every_expert() -> Result<()> {
    check(RouterUnitScoring {
        weighting: RouterUnitWeighting::Probability,
        scale: 1.0,
    })
}

/// The bias moves the selection while the weights stay unbiased sigmoids.
#[test]
fn selects_by_biased_sigmoid_and_weighs_by_sigmoid() -> Result<()> {
    check(RouterUnitScoring {
        weighting: RouterUnitWeighting::SigmoidBiased,
        scale: 2.5,
    })
}

#[test]
fn scales_probability_weights() -> Result<()> {
    check(RouterUnitScoring {
        weighting: RouterUnitWeighting::Probability,
        scale: 16.0,
    })
}

fn check(scoring: RouterUnitScoring) -> Result<()> {
    let runtime = Runtime::new()?;
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
    let router = RouterUnitTopK::compile_scored(
        &runtime.compiler,
        spec,
        scoring,
        sigmoid.then_some(&device_bias),
    )?;
    let device_scores = runtime.copy(&scores)?;
    let mut selected = runtime.pool.allocate::<u32>(&runtime.stream, TOKENS * TOP_K)?;
    let mut weights = runtime.pool.allocate::<bf16>(&runtime.stream, TOKENS * TOP_K)?;
    router.execute(&runtime.stream, &device_scores, &mut selected, &mut weights)?;
    let (selected, weights) = (runtime.read(&selected)?, runtime.read(&weights)?);
    for token in 0..TOKENS {
        let logits = scores[token * EXPERTS..(token + 1) * EXPERTS]
            .iter()
            .map(|value| value.to_f32())
            .collect::<Vec<_>>();
        let (indices, expected) = reference(&logits, &bias, scoring);
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
    Ok(())
}

/// Experts in descending logit order (lower index first on ties) and their
/// weights.
fn reference(logits: &[f32], bias: &[f32], scoring: RouterUnitScoring) -> (Vec<usize>, Vec<f32>) {
    let gate = |index: usize| (1.0 + (-logits[index]).exp()).recip();
    let choice = |index: usize| match scoring.weighting {
        RouterUnitWeighting::SigmoidBiased => gate(index) + bias[index],
        RouterUnitWeighting::Renormalized | RouterUnitWeighting::Probability => logits[index],
    };
    let mut order = (0..logits.len()).collect::<Vec<_>>();
    order.sort_by(|&a, &b| choice(b).total_cmp(&choice(a)).then(a.cmp(&b)));
    order.truncate(TOP_K);
    if scoring.weighting == RouterUnitWeighting::SigmoidBiased {
        let sum = order.iter().map(|&index| gate(index)).sum::<f32>();
        let weights = order.iter().map(|&index| gate(index) / sum * scoring.scale).collect();
        return (order, weights);
    }
    let maximum = logits[order[0]];
    let exponent = |index: usize| (logits[index] - maximum).exp();
    let denominator = match scoring.weighting {
        RouterUnitWeighting::Renormalized => order.iter().map(|&index| exponent(index)).sum(),
        RouterUnitWeighting::Probability | RouterUnitWeighting::SigmoidBiased => {
            (0..logits.len()).map(exponent).sum::<f32>()
        },
    };
    let weights = order
        .iter()
        .map(|&index| exponent(index) / denominator * scoring.scale)
        .collect();
    (order, weights)
}

struct Runtime {
    context: Context,
    compiler: Compiler,
    pool: MemoryPool,
    stream: Stream,
}

impl Runtime {
    fn new() -> Result<Self> {
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

    fn copy<T: DeviceElement + Copy>(&self, values: &[T]) -> Result<DeviceBuffer<T>> {
        let mut host = self.context.allocate_pinned::<T>(values.len())?;
        host.copy_from_slice(values)?;
        let mut device = self.pool.allocate::<T>(&self.stream, values.len())?;
        self.stream.copy_to_device(&mut host, &mut device)?;
        self.stream.synchronize()?;
        Ok(device)
    }

    fn read<T: DeviceElement + Copy>(&self, device: &DeviceBuffer<T>) -> Result<Vec<T>> {
        let mut host = self.context.allocate_pinned::<T>(device.len())?;
        self.stream.copy_to_host(device, &mut host)?;
        self.stream.synchronize()?;
        Ok(host.to_vec()?)
    }
}
