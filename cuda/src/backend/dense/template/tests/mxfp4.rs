use std::{fs, path::Path};

use mircuda::bf16;
use models::weights::{
    BlockQuantization, LogicalTensorRole, TensorBinding, TensorInfo, TensorPacking, TensorStorage,
};
use runtime::kv::{BlockId, BlockTable, CacheConfig, KvCacheDType, KvStorageSpec};

use super::{copy, read};
use crate::{
    CudaBackend, CudaConfig, DenseDownSource, DenseGateUpSource, DenseOutputSource, DenseQkvSource,
    DenseSwiGluConfig, DenseWeightSource, GatedActivation, MxFp4CheckpointWeight, ProjectionFormat,
    Result, kernels::QkvNormalization,
};

#[test]
fn executes_complete_dense_layer_from_mxfp4_checkpoint_blocks() -> Result<()> {
    let path = std::env::temp_dir().join(format!("libmir-cuda-mxfp4-dense-{}.bin", process_id()));
    let infos = fixture(&path)?;
    let backend = CudaBackend::new(CudaConfig::default())?;
    let mut upload = backend.begin_tensor_upload();
    for info in &infos {
        upload.enqueue(info)?;
    }
    let tensors = upload.finish()?;
    let weight = MxFp4CheckpointWeight::load_binding(&tensors, &binding())?;
    let norm = tensors.get("norm").ok_or_else(|| crate::Error::MissingTensor("norm".into()))?;
    let config = config();
    let template = backend.prepare_dense_swiglu_layer_template(
        config,
        DenseWeightSource {
            input_norm: norm,
            qkv: DenseQkvSource::MxFp4([&weight, &weight, &weight]),
            query_norm: None,
            key_norm: None,
            output: DenseOutputSource::MxFp4(&weight),
            post_attention_norm: norm,
            gate_up: DenseGateUpSource::MxFp4 { gate: &weight, up: &weight },
            down: DenseDownSource::MxFp4(&weight),
        },
    )?;
    let input = copy(&backend, &[bf16::ONE; 32])?;
    let mut output = backend.inner.pool.allocate_zeroed(&backend.inner.stream, 32)?;
    let cache = backend.prepare_paged_kv(0, config.attention.cache)?;
    let mut state = template.instantiate_with_cache(&input, &output, cache)?;
    let mut prefill = template.instantiate_prefill(1)?;
    let mut table = BlockTable::with_block_size(16);
    table.push(BlockId(0));
    table.set_token_len(1);
    let mut batch = backend.prepare_paged_prefill_batch(config.attention.cache, 1, 1, 1)?;
    batch.prepare(&[&table], &[0], &[1])?;
    prefill.execute_batch(&mut state, &input, template.weights(), &batch, &mut output)?;
    let output = read(&backend, &output)?;
    assert!(output.iter().all(|value| value.to_f32().is_finite()));
    assert!(output.iter().all(|value| value.to_f32() > 1.0));
    fs::remove_file(path)?;
    Ok(())
}

fn config() -> DenseSwiGluConfig {
    DenseSwiGluConfig {
        attention: crate::DecodeAttentionConfig {
            layer: 0,
            hidden_size: 32,
            query_heads: 1,
            rotary_dim: 32,
            rope_pairing_dim: 32,
            rope_theta: 10_000.0,
            rms_norm_epsilon: 1.0e-6,
            attention_scale: 32.0_f32.sqrt().recip(),
            projection_format: ProjectionFormat::MxFp4,
            qkv_normalization: QkvNormalization::NONE,
            sliding_window: None,
            max_sequence_blocks: 1,
            cache: KvStorageSpec::new(
                CacheConfig {
                    block_size: 16,
                    block_count: 1,
                    dtype: KvCacheDType::BFloat16,
                },
                1,
                32,
            ),
        },
        intermediate_size: 32,
        activation: GatedActivation::Silu,
    }
}

fn fixture(path: &Path) -> Result<[TensorInfo; 3]> {
    let mut bytes = vec![0x22_u8; 32 * 16];
    let weight_end = u64::try_from(bytes.len())?;
    bytes.extend([127_u8; 32]);
    let scale_end = u64::try_from(bytes.len())?;
    for _ in 0..32 {
        bytes.extend_from_slice(&bf16::ONE.to_bits().to_le_bytes());
    }
    let end = u64::try_from(bytes.len())?;
    fs::write(path, bytes)?;
    Ok([
        info("weight", path, "U8", vec![32, 1, 16], 0, weight_end),
        info("scales", path, "U8", vec![32, 1], weight_end, scale_end),
        info("norm", path, "BF16", vec![32], scale_end, end),
    ])
}

fn binding() -> TensorBinding {
    TensorBinding {
        role: LogicalTensorRole::Output,
        source: "weight".into(),
        shape: vec![32, 1, 16],
        logical_shape: Some(vec![32, 32]),
        transforms: Vec::new(),
        storage: TensorStorage::BlockQuantized {
            format: BlockQuantization::MXFP4,
            scales: "scales".into(),
            global_scale: None,
            input_scale: None,
            bias: None,
            packing: TensorPacking::Separate,
        },
    }
}

fn info(
    name: &str,
    path: &Path,
    dtype: &str,
    shape: Vec<usize>,
    start: u64,
    end: u64,
) -> TensorInfo {
    TensorInfo {
        name: name.into(),
        file: path.to_path_buf(),
        dtype: dtype.into(),
        shape,
        data_start: 0,
        data_offsets: [start, end],
    }
}

fn process_id() -> u32 {
    std::process::id()
}
