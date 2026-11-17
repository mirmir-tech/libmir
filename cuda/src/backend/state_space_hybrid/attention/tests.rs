use std::{fs, path::PathBuf};

use mircuda::{DeviceBuffer, DeviceElement, bf16};
use models::weights::TensorInfo;
use runtime::kv::{BlockId, BlockTable, CacheConfig, KvCacheDType, KvStorageSpec, KvWritePlan};
use uuid::Uuid;

use super::{AttentionWeights, CudaAttentionHybridLayer, Projections};
use crate::{
    CudaBackend, CudaConfig, CudaTensorSet, DecodeAttentionConfig, Error, ProjectionFormat, Result,
    backend::feed_forward::LayerNormConfig, kernels::QkvNormalization,
};

const HIDDEN: usize = 512;
const QUERY_HEADS: usize = 4;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 128;
/// `1 / sqrt(HEAD_DIM)`.
const ATTENTION_SCALE: f32 = 0.088_388_35;
const NAMES: [&str; 5] = ["norm", "query", "key", "value", "output"];

/// A packed decode step built after the model's pages were resized reads
/// and writes the new pages: it matches the single-sequence step in blocks
/// the cache the layer was loaded with does not have.
#[test]
fn packed_decode_follows_resized_pages() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let weights = Weights::new()?;
    let tensors = weights.upload(&backend)?;
    let layer = layer(&backend, &tensors)?;
    let resized = layer.storage(cache(8));
    let pages = backend.prepare_paged_kv(0, resized)?;
    let input = copy(&backend, &values(HIDDEN, 3))?;

    let single_table = table(6);
    let mut single_state = layer.prepare_state_with_cache(pages.clone())?;
    let mut single = layer.prepare(1)?;
    let mut expected = backend.inner.pool.allocate(&backend.inner.stream, HIDDEN)?;
    single.execute(
        &input,
        &mut single_state,
        &KvWritePlan::prefill(Uuid::nil(), 0, &single_table, 0, 1)?,
        &single_table,
        0,
        &mut expected,
    )?;

    let packed_table = table(7);
    let packed_state = layer.prepare_state_with_cache(pages)?;
    let mut paging = backend.prepare_paged_decode_batch(resized, 2, 1)?;
    paging.prepare(&[&packed_table])?;
    let mut packed = layer.prepare(1)?;
    packed.prepare_rows(&packed_state, &paging)?;
    let mut actual = backend.inner.pool.allocate(&backend.inner.stream, HIDDEN)?;
    packed.execute_prepared_packed(&input, &paging, &mut actual)?;

    let expected = read(&backend, &expected)?;
    let actual = read(&backend, &actual)?;
    let difference = actual
        .iter()
        .zip(&expected)
        .map(|(left, right)| (left.to_f32() - right.to_f32()).abs())
        .fold(0.0_f32, f32::max);
    assert!(expected.iter().any(|value| value.to_f32() != 0.0));
    assert!(difference < 0.05, "packed and single decode differ by {difference}");
    Ok(())
}

fn layer(backend: &CudaBackend, tensors: &CudaTensorSet) -> Result<CudaAttentionHybridLayer> {
    let get = |name: &str| tensors.get(name).ok_or_else(|| Error::MissingTensor(name.into()));
    let config = DecodeAttentionConfig {
        layer: 0,
        hidden_size: HIDDEN,
        query_heads: QUERY_HEADS,
        rotary: None,
        rms_norm_epsilon: 1.0e-5,
        attention_scale: ATTENTION_SCALE,
        projection_format: ProjectionFormat::Bf16,
        qkv_normalization: QkvNormalization::NONE,
        sliding_window: None,
        max_sequence_blocks: 2,
        cache: KvStorageSpec::new(cache(2), KV_HEADS, HEAD_DIM),
    };
    let qkv = backend.pack_bf16_linears([get("query")?, get("key")?, get("value")?])?;
    Ok(CudaAttentionHybridLayer {
        backend: backend.clone(),
        norm: LayerNormConfig {
            hidden_size: HIDDEN,
            rms_norm_epsilon: config.rms_norm_epsilon,
            norm_weight_shift: 0.0,
        },
        config,
        weights: AttentionWeights {
            input_norm: get("norm")?.clone(),
            projections: Projections::Bf16 { qkv, output: get("output")?.clone() },
            unused_norm: backend.inner.pool.allocate_zeroed(&backend.inner.stream, HEAD_DIM)?,
        },
        feed_forward: None,
    })
}

const fn cache(block_count: u32) -> CacheConfig {
    CacheConfig {
        block_size: 16,
        block_count,
        dtype: KvCacheDType::BFloat16,
    }
}

fn table(block: u32) -> BlockTable {
    let mut table = BlockTable::with_block_size(16);
    table.push(BlockId(block));
    table.set_token_len(1);
    table
}

/// Small varied values, so projections and attention are not degenerate.
fn values(count: usize, seed: usize) -> Vec<bf16> {
    (0..count)
        .map(|index| {
            let step = u16::try_from((index * 7 + seed * 13) % 17).unwrap_or_default();
            bf16::from_f32(f32::from(step) / 128.0 - 0.0625)
        })
        .collect()
}

/// BF16 weights of one layer in a temporary file the upload reads.
struct Weights {
    path: PathBuf,
    infos: Vec<TensorInfo>,
}

impl Weights {
    fn new() -> Result<Self> {
        let path = std::env::temp_dir()
            .join(format!("libmir-cuda-hybrid-attention-{}.bin", Uuid::new_v4()));
        let shapes = [
            vec![HIDDEN],
            vec![QUERY_HEADS * HEAD_DIM, HIDDEN],
            vec![KV_HEADS * HEAD_DIM, HIDDEN],
            vec![KV_HEADS * HEAD_DIM, HIDDEN],
            vec![HIDDEN, QUERY_HEADS * HEAD_DIM],
        ];
        let mut bytes = Vec::new();
        let mut infos = Vec::new();
        for (seed, (name, shape)) in NAMES.iter().zip(shapes).enumerate() {
            let start = u64::try_from(bytes.len())?;
            let count = shape.iter().product();
            let data = if seed == 0 {
                vec![bf16::from_f32(1.0); count]
            } else {
                values(count, seed)
            };
            for value in data {
                bytes.extend_from_slice(&value.to_bits().to_le_bytes());
            }
            infos.push(TensorInfo {
                name: (*name).into(),
                file: path.clone(),
                dtype: "BF16".into(),
                shape,
                data_start: 0,
                data_offsets: [start, u64::try_from(bytes.len())?],
            });
        }
        fs::write(&path, bytes)?;
        Ok(Self { path, infos })
    }

    fn upload(&self, backend: &CudaBackend) -> Result<CudaTensorSet> {
        let mut upload = backend.begin_tensor_upload();
        for info in &self.infos {
            upload.enqueue(info)?;
        }
        upload.finish()
    }
}

impl Drop for Weights {
    fn drop(&mut self) {
        let _removed = fs::remove_file(&self.path);
    }
}

fn read<T: DeviceElement>(backend: &CudaBackend, source: &DeviceBuffer<T>) -> Result<Vec<T>> {
    backend.synchronize()?;
    let mut host = backend.inner.context.allocate_pinned::<T>(source.len())?;
    backend.inner.stream.copy_to_host(source, &mut host)?;
    Ok(host.to_vec()?)
}

fn copy<T: DeviceElement>(backend: &CudaBackend, values: &[T]) -> Result<DeviceBuffer<T>> {
    let mut host = backend.inner.context.allocate_pinned::<T>(values.len())?;
    host.copy_from_slice(values)?;
    let mut device = backend.inner.pool.allocate::<T>(&backend.inner.stream, values.len())?;
    backend.inner.stream.copy_to_device(&mut host, &mut device)?;
    backend.synchronize()?;
    Ok(device)
}
