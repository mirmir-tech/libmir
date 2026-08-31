use std::path::Path;

use mircuda::{DenseVectorPlan, DenseVectorSpec, bf16};
use models::{
    execution::DecoderExecutionContract,
    layout::{DecoderConfig, ModelLayout},
    weights::{TensorCatalog, TensorStorage},
};

use super::metrics::Metrics;
use crate::{
    CudaBackend, CudaConfig, Error, Result,
    kernels::{BlockFp8LinearKernels, BlockFp8LinearSpec, Fp8OutputSpec, Fp8RefinementKernels},
};

const VECTORS: usize = 8;

#[test]
#[allow(clippy::print_stderr)]
fn checkpoint_dense_output_head_preserves_rank_and_distribution()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let Some(root) = std::env::var_os("LIBMIR_CUDA_GATE_OUTPUT_TENSOR") else {
        return Ok(());
    };
    let layout = ModelLayout::inspect(Path::new(&root))?;
    let decoder = DecoderConfig::from_layout(&layout)?;
    let catalog = TensorCatalog::from_layout(&layout)?;
    let contract = DecoderExecutionContract::discover(&layout, &decoder, &catalog)?;
    let binding = contract.bindings.decoder_boundary()?.output;
    if !matches!(binding.storage, TensorStorage::Dense { .. }) || !binding.transforms.is_empty() {
        eprintln!(
            "output tensor gate skipped checkpoint-native {:?} storage for {}",
            binding.storage, binding.source
        );
        return Ok(());
    }
    let info = catalog
        .tensors
        .iter()
        .find(|tensor| tensor.name == binding.source)
        .ok_or_else(|| Error::MissingTensor(binding.source.clone()))?;
    let backend = CudaBackend::new(CudaConfig::default())?;
    let cast = backend.prepare_dense_cast()?;
    let mut upload = backend.begin_tensor_upload();
    upload.enqueue_as_bf16(info, &cast)?;
    let tensors = upload.finish()?;
    let weight =
        tensors
            .get(&binding.source)
            .and_then(crate::CudaTensor::as_bf16)
            .ok_or_else(|| Error::DTypeMismatch {
                name: binding.source.clone(),
                expected: "BF16",
            })?;
    let metrics = compare(&backend, weight, decoder.hidden_size, decoder.vocab_size)?;
    eprintln!(
        "output tensor gate {} {}x{}: {}",
        binding.source,
        decoder.hidden_size,
        decoder.vocab_size,
        report(&metrics),
    );
    metrics.validate("fp8-block-refined output tensor");
    Ok(())
}

fn compare(
    backend: &CudaBackend,
    exact_weight: &mircuda::DeviceBuffer<bf16>,
    input_features: usize,
    output_features: usize,
) -> Result<Metrics> {
    let spec = BlockFp8LinearSpec::new(input_features, output_features)?;
    let kernels = BlockFp8LinearKernels::compile(&backend.inner.compiler, spec)?;
    let mut weight = backend
        .inner
        .pool
        .allocate::<u8>(&backend.inner.stream, spec.weight_elements()?)?;
    let mut scales = backend
        .inner
        .pool
        .allocate::<f32>(&backend.inner.stream, spec.scale_elements()?)?;
    kernels.quantize(&backend.inner.stream, exact_weight, &mut weight, &mut scales)?;
    let refinement = Fp8RefinementKernels::compile(
        &backend.inner.compiler,
        Fp8OutputSpec::new_refinement(input_features, output_features)?,
    )?;
    let workspace = Fp8RefinementKernels::workspace_elements(output_features)?;
    let mut first = backend.inner.pool.allocate::<u64>(&backend.inner.stream, workspace)?;
    let mut second = backend.inner.pool.allocate::<u64>(&backend.inner.stream, workspace)?;
    let mut exact = DenseVectorPlan::new(
        &backend.inner.context,
        &backend.inner.stream,
        DenseVectorSpec::new(output_features, input_features)?,
    )?;
    let mut input = backend.inner.pool.allocate::<bf16>(&backend.inner.stream, input_features)?;
    let mut exact_logits =
        backend.inner.pool.allocate::<bf16>(&backend.inner.stream, output_features)?;
    let mut actual_logits =
        backend.inner.pool.allocate::<bf16>(&backend.inner.stream, output_features)?;
    let mut metrics = Metrics::default();
    for seed in 0..VECTORS {
        upload_input(backend, &mut input, seed)?;
        exact.execute(&backend.inner.stream, &input, exact_weight, &mut exact_logits, 1.0, 0.0)?;
        kernels.project(&backend.inner.stream, &input, &weight, &scales, &mut actual_logits)?;
        refinement.execute(
            &backend.inner.stream,
            &input,
            exact_weight,
            &mut actual_logits,
            &mut first,
            &mut second,
        )?;
        metrics.observe(
            &super::super::super::read(backend, &exact_logits)?,
            &super::super::super::read(backend, &actual_logits)?,
        );
    }
    Ok(metrics)
}

fn upload_input(
    backend: &CudaBackend,
    input: &mut mircuda::DeviceBuffer<bf16>,
    seed: usize,
) -> Result<()> {
    let mut host = backend.inner.context.allocate_pinned::<bf16>(input.len())?;
    let values = (0..input.len())
        .map(|index| {
            let sample =
                index.wrapping_mul(1_103_515_245).wrapping_add(seed.wrapping_mul(12_345)) % 2_001;
            let centered = i16::try_from(sample)? - 1_000;
            Ok(bf16::from_f32(f32::from(centered) / 577.0))
        })
        .collect::<Result<Vec<_>>>()?;
    host.copy_from_slice(&values)?;
    backend.inner.stream.copy_to_device(&mut host, input)?;
    Ok(())
}

#[allow(clippy::cast_precision_loss)]
fn report(metrics: &Metrics) -> String {
    format!(
        "steps={} top1={:.3}% top10_overlap={:.3}% nrmse={:.6} max_abs={:.6} mean_kl={:.6}",
        metrics.steps,
        metrics.top1 as f64 * 100.0 / metrics.steps as f64,
        metrics.topk_overlap as f64 * 10.0 / metrics.steps as f64,
        (metrics.squared_error / metrics.squared_reference.max(f64::EPSILON)).sqrt(),
        metrics.maximum_error,
        metrics.kl_divergence / metrics.steps as f64,
    )
}
